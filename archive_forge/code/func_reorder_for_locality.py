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
def reorder_for_locality(graph: torch.fx.Graph):

    def visit(other_node):
        if other_node.op == 'call_function' and other_node.target != operator.getitem and all((n in seen_nodes for n in other_node.users)):
            node.prepend(other_node)
    seen_nodes = set()
    first_copy = next((node for node in graph.nodes if node.op == 'call_function' and node.target == torch.ops.aten.copy_.default), None)
    past_mutating_epilogue = True if first_copy is None else False
    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        if not past_mutating_epilogue:
            past_mutating_epilogue = node is first_copy
            continue
        torch.fx.map_arg((node.args, node.kwargs), visit)