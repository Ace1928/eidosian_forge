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
def get_cpu_indeg_count(self, graph: fx.Graph) -> Dict[fx.Node, int]:
    """
        Get the number of cpu inputs to a node
        """
    cpu_indeg: Dict[fx.Node, int] = Counter()
    for node in graph.nodes:
        cpu_count = 0

        def add_cpu_inp(node):
            nonlocal cpu_count
            device = self.get_node_device(node)
            cpu_count += device is not None and device.type == 'cpu'
        pytree.tree_map_only(fx.Node, add_cpu_inp, (node.args, node.kwargs))
        if cpu_count:
            cpu_indeg[node] = cpu_count
    return cpu_indeg