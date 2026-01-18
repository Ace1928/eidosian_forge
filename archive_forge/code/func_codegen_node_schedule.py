from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
def codegen_node_schedule(self, node_schedule, numel, reduction_numel):
    tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
    reduction_hint_val, mutations, index_dtype = self.get_kernel_args(node_schedule, numel, reduction_numel)
    kernel = TritonKernel(*tiled_groups, reduction_hint=reduction_hint_val, mutations=mutations, index_dtype=index_dtype)
    self.codegen_node_schedule_with_kernel(node_schedule, kernel)
    with V.set_kernel_handler(kernel):
        src_code = kernel.codegen_kernel()
        for node in node_schedule:
            if node not in (EnableReduction, DisableReduction):
                node.mark_run()
    kernel_name = self.define_kernel(src_code, node_schedule)
    log.debug('Generating kernel code with kernel_name: %s', kernel_name)
    self.codegen_comment(node_schedule)
    kernel.call_kernel(kernel_name)
    kernel.codegen_nan_check()
    V.graph.removed_buffers |= kernel.removed_buffers
    V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
    if config.warn_mix_layout:
        kernel.warn_mix_layout(kernel_name)
    if V.graph.wrapper_code.supports_intermediate_hooks and config.generate_intermediate_hooks:
        live_outs = kernel.args.live_output_buffers()
        for node in node_schedule:
            if not isinstance(node, scheduler.BaseSchedulerNode):
                continue
            name = node.get_name()
            if name not in live_outs:
                continue
            origin_node = node.node.get_origin_node()
            if origin_node is not None:
                counters['inductor']['intermediate_hooks'] += 1
                V.graph.wrapper_code.writeline(f'run_intermediate_hooks({origin_node.name!r}, {name})')
    self.scheduler.free_buffers()