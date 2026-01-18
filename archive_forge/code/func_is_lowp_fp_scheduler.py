import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
def is_lowp_fp_scheduler(self, scheduler_node: SchedulerNode):
    if not isinstance(scheduler_node._body, ir.LoopBody):
        return True
    _lowp_fp_type: Optional[torch.dtype] = None
    DataTypePropagation.propagate_scheduler_node(scheduler_node)
    sub_blocks = [scheduler_node._body.root_block] + list(scheduler_node._body.subblocks.values())
    for sub_block in sub_blocks:
        for _node in sub_block.graph.nodes:
            if _node.op == 'placeholder' or _node.target in ('get_index', 'index_expr'):
                continue
            if _node.target not in ['load', 'store', 'abs', 'neg', 'output']:
                return False
            if hasattr(_node, 'meta') and _node.meta:
                assert OptimizationContext.key in _node.meta
                opt_ctx: OptimizationContext = _node.meta[OptimizationContext.key]
                if not opt_ctx.dtype or opt_ctx.dtype not in DTYPE_LOWP_FP:
                    return False
                if _lowp_fp_type:
                    assert _lowp_fp_type == opt_ctx.dtype, 'scheduler node do not support bf16/fp16 mix'
                else:
                    _lowp_fp_type = opt_ctx.dtype
            else:
                return False
    scheduler_node._lowp_fp_type = _lowp_fp_type
    return True