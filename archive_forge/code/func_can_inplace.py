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