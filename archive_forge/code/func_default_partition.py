from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
def default_partition(joint_module: fx.GraphModule, _joint_inputs, *, num_fwd_outputs) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the :attr:`joint_module` in a manner that closely resembles the
    behavior observed in the original ``.forward()`` and ``.backward()`` of the
    callable, i.e., the resulting forward graph contains those operators that
    are executed in the original ``.forward()`` callable passed to
    :func:`aot_function`.

    The default partitioner collects the operators that are between the forward
    inputs and the forward outputs. This helps in finding the tensors which have
    to be stashed for the backward pass. These stashed tensors become the output
    of the generated forward graph. The remaining operators are then placed in
    the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
    if has_recomputable_ops(joint_module):
        return min_cut_rematerialization_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, inputs, fwd_outputs)
    forward_node_names = {node.name for node in forward_only_graph.nodes if node.op != 'output'}
    saved_values = []
    saved_sym_nodes = []
    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        elif 'tensor_meta' not in node.meta and node.op == 'call_function':
            users = node.users
            assert all((user.target == operator.getitem for user in users))
            for user in users:
                saved_values.append(user)
        else:
            backward_usages = [n for n in node.users if n.name not in forward_node_names]
            if 'tensor_meta' in node.meta and all((is_sym_node(n) for n in backward_usages)):
                for user in backward_usages:
                    saved_sym_nodes.append(user)
            else:
                saved_values.append(node)
    saved_values = list({k: None for k in saved_values}.keys())
    saved_sym_nodes = list({k: None for k in saved_sym_nodes}.keys())
    return _extract_fwd_bwd_modules(joint_module, saved_values, saved_sym_nodes=saved_sym_nodes, num_fwd_outputs=num_fwd_outputs)