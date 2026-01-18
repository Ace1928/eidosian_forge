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
def gen_loops(loops: List[LoopLevel], in_reduction=False):
    with contextlib.ExitStack() as stack_outer:
        if loops:
            loop = loops[0]
            if loop.is_reduction() and (not in_reduction):
                reduction_prefix = get_reduction_code_buffer(loops, is_suffix=False)
                if reduction_prefix:
                    stack_outer.enter_context(code.indent())
                code.splice(reduction_prefix)
            if loop_nest.is_reduction_only() and loop.parallel:
                worksharing.parallel(threads)
        for loop in loops:
            gen_loop(loop, in_reduction)
        if loops:
            loop = loops[0]
            if loop_nest.is_reduction_only() and loop.parallel:
                worksharing.close()
            if loop.is_reduction() and (not in_reduction):
                code.splice(get_reduction_code_buffer(loops, is_suffix=True))