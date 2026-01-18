import copy
import logging
from typing import List, Optional
import torch
import torch.nn as nn
from torch._dynamo.utils import detect_fake_mode
from torch._utils_internal import print_graph
from torch.fx.experimental.optimization import (
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from .. import config
from ..fx_utils import matches_module_function_pattern
from ..pattern_matcher import (
from ..utils import is_cpu_device
from .group_batch_fusion import group_batch_fusion_passes
from .misc_patterns import numpy_compat_normalization
def pre_grad_passes(gm: torch.fx.GraphModule, example_inputs):
    """
    Apply passes on the input FX graph using Torch IR.

    WARNING:
    The IR before grad is not functional or normalized, so it is harder
    to write passes on this IR.  Passes must be safe with respect to
    aliasing and mutation and need to handle all possible arg schemas.

    Consider adding a new pass to post_grad.py or joint_graph.py which
    are after functionalization and normalization.
    """
    if config.pattern_matcher:
        lazy_init()
        gm = fuse_fx(gm, example_inputs)
        numpy_compat_normalization(gm.graph)
        group_batch_fusion_passes(gm.graph, pre_grad=True)
        print_graph(gm.graph, 'Before split cat in pre grad pass.')
        for pattern_matcher_pass in pattern_matcher_passes:
            pattern_matcher_pass.apply(gm.graph)
            print_graph(gm.graph, f'Apply split cat pattern matcher {pattern_matcher_pass.__class__.__name__} in pre grad.')
    if config.pre_grad_custom_pass is not None:
        config.pre_grad_custom_pass(gm.graph)
    stable_topological_sort(gm.graph)
    gm.graph.lint()
    gm.recompile()
    print_graph(gm.graph, 'Aftre recompile in pre grad pass.')
    return gm