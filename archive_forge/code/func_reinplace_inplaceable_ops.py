import functools
import itertools
import logging
import operator
from collections import Counter, defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Union
from sympy import Expr
import torch
import torch._inductor as inductor
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp import register_decomposition
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional
from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype
from torch._utils_internal import print_graph
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.fx.immutable_collections import immutable_dict
from .. import config, inductor_prims, ir, pattern_matcher
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage
from ..lowering import (
from ..pattern_matcher import (
from ..utils import decode_device, is_pointwise_use
from ..virtualized import V
from .group_batch_fusion import group_batch_fusion_passes
def reinplace_inplaceable_ops(graph):
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    This condition is slightly different for graph inputs where they can only
    be inplaced if the above condition is true and there's a copy_ in the
    epilogue that signals that the caller wants to observe the mutation.
    """
    copy_args_to_copy_nodes = {}
    foreach_node_to_copy_nodes = defaultdict(list)
    mutated_inputs = set()
    storage_to_nodes = defaultdict(list)
    node_order: Dict[Any, int] = {}
    for i, node in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage_to_nodes[get_node_storage(node)].append(node)
        if node.target == aten.copy_.default:
            dst = node.args[0]
            src = node.args[1]
            if src.target == operator.getitem and (src.args[0].target == triton_kernel_wrapper_functional and src.args[0].kwargs['kwargs'][src.args[1]] == node.args[0] or src.args[0].target in inplaceable_foreach_ops):
                src = src.args[0]
            copy_args_to_copy_nodes[dst, src] = node
            assert node.args[0].op == 'placeholder'
            mutated_inputs.add(node.args[0])

    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node):
        node_loc = node_order[node]
        for view in shared_view_nodes:
            for user in view.users:
                if node_order[user] <= node_loc:
                    continue
                if copy_node == user:
                    continue
                return True
        return False

    def can_inplace(node, mutated_arg):
        if isinstance(mutated_arg, (list, tuple)):
            return all((can_inplace(node, arg) for arg in mutated_arg))
        if get_node_storage(mutated_arg) is None:
            return False
        shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]
        if mutated_arg.op == 'placeholder':
            if not (copy_node := copy_args_to_copy_nodes.get((mutated_arg, node), False)):
                return False
            if any_use_of_views_after_node(node, shared_view_nodes, copy_node=copy_node):
                return False
            return True
        elif any((view.op == 'placeholder' for view in shared_view_nodes)):
            return False
        else:
            return not any_use_of_views_after_node(node, shared_view_nodes, copy_node=None)
    for node in graph.nodes:
        if (inplaceable_op := inplaceable_ops.get(node.target, None)) is not None:
            mutated_arg = node.args[inplaceable_op.mutated_arg]
            if can_inplace(node, mutated_arg):
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    graph.erase_node(copy_node)
                node.target = inplaceable_op.inplace_op
        elif node.target in inplaceable_triton_ops:
            tensors_to_clone = []
            for arg in node.kwargs['tensors_to_clone']:
                assert arg in node.kwargs['kwargs']
                mutated_arg = node.kwargs['kwargs'][arg]
                if can_inplace(node, mutated_arg):
                    copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                    if copy_node is not None:
                        graph.erase_node(copy_node)
                else:
                    tensors_to_clone.append(arg)
            kwargs = dict(node.kwargs)
            kwargs['tensors_to_clone'] = tensors_to_clone
            node.kwargs = immutable_dict(kwargs)
        elif (inplaceable_op := inplaceable_foreach_ops.get(node.target, None)) is not None:
            mutated_args = node.args[inplaceable_op.mutated_arg]
            if not all(((arg, node) in copy_args_to_copy_nodes for arg in mutated_args)):
                continue
            if can_inplace(node, mutated_args):
                for arg in mutated_args:
                    copy_node = copy_args_to_copy_nodes[arg, node]
                    graph.erase_node(copy_node)
                node.target = inplaceable_op.inplace_op