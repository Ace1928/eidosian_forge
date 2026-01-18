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
def is_valid_splitwithsizes_cat(match):
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = (split_nodes[0], cat_nodes[0])
    if get_arg_value(split_node, 2, 'dim') != get_arg_value(cat_node, 1, 'dim'):
        return False
    get_item_args = {get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes}
    assert None not in get_item_args
    split_sizes = get_arg_value(split_node, 1, 'split_sizes')
    if get_item_args != set(range(len(split_sizes))):
        return False
    cat_items_args_order = [get_arg_value(item_node, 1) for item_node in get_arg_value(cat_node, 0)]
    if cat_items_args_order != list(range(len(split_sizes))):
        return False
    return True