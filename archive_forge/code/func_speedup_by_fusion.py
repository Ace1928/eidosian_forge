import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
def speedup_by_fusion(self, node1, node2):
    """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """
    if not config.benchmark_fusion:
        return True
    if node1.is_template():
        return True
    node_list_1 = node1.get_nodes()
    device = node_list_1[0].get_device()
    if device.type == 'cpu':
        return True
    node_list_2 = node2.get_nodes()
    node_list_fused = node_list_1 + node_list_2
    if any((hasattr(n.node, 'data') and hasattr(n.node.data, 'scatter_mode') and (n.node.data.scatter_mode == 'atomic_add') for n in node_list_fused)):
        return True
    from triton.compiler.errors import CompilationError
    try:
        ms1, path1 = self.benchmark_fused_nodes(node_list_1)
        if math.isinf(ms1):
            fusion_log.debug('cannot fuse (benchmark): register spilling of the first kernel')
            return False
        ms2, path2 = self.benchmark_fused_nodes(node_list_2)
        if math.isinf(ms2):
            fusion_log.debug('cannot fuse (benchmark): register spilling of the second kernel')
            return False
        ms_fused, path_fused = self.benchmark_fused_nodes(node_list_fused)
        if math.isinf(ms_fused):
            fusion_log.debug('cannot fuse (benchmark): register spilling of the fused kernel')
            return False
    except CompilationError as e:
        if 'Loop-carried variable' in str(e):
            return True
        else:
            raise
    if fusion_log.isEnabledFor(logging.DEBUG):
        if ms_fused < ms1 + ms2:
            fusion_log.debug('can fuse (benchmark): fusing %s with %s cause %sx speedup', node1.get_names(), node2.get_names(), green_text(f'{(ms1 + ms2) / ms_fused:.3f}'))
        else:
            fusion_log.debug('cannot fuse (benchmark): fusing %s with %s cause %sx slowdown', node1.get_names(), node2.get_names(), red_text(f'{ms_fused / (ms1 + ms2):.3f}'))
    if is_metric_table_enabled('slow_fusion') and ms_fused >= ms1 + ms2 and ((path1, path2) not in self.logged_slow_fusion):
        self.logged_slow_fusion.add((path1, path2))
        get_metric_table('slow_fusion').add_row(lambda: {'kernel1_path': path1, 'kernel1_latency': ms1, 'kernel2_path': path2, 'kernel2_latency': ms2, 'fused_kernel_path': path_fused, 'fused_kernel_latency': ms_fused, 'slow_down_ratio': ms_fused / (ms1 + ms2)})
    return ms_fused < ms1 + ms2