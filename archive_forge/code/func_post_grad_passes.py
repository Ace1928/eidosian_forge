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
def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    if config.dce:
        gm.graph.eliminate_dead_code()
    if is_inference and config.reorder_for_locality:
        reorder_for_locality(gm.graph)
    fake_tensor_updater = FakeTensorUpdater(gm.graph)
    if config.post_grad_custom_pre_pass is not None:
        config.post_grad_custom_pre_pass(gm.graph)
    if config.pattern_matcher:
        lazy_init()
        group_batch_fusion_passes(gm.graph, pre_grad=False)
        remove_noop_ops(gm.graph)
        print_graph(gm.graph, 'Before split cat in post grad pass.')
        for patterns in pass_patterns:
            patterns.apply(gm.graph)
            print_graph(gm.graph, f'Apply split cat pattern matcher {patterns.__class__.__name__} in post grad.')
        if is_inference:
            inference_patterns.apply(gm.graph)
    if config.post_grad_custom_post_pass is not None:
        config.post_grad_custom_post_pass(gm.graph)
    stable_topological_sort(gm.graph)
    move_constructors_to_cuda(gm.graph)
    fake_tensor_updater.incremental_update()
    reinplace_inplaceable_ops(gm.graph)
    gm.recompile()
    gm.graph.lint()
    print_graph(gm.graph, 'Aftre recompile in post grad pass.')