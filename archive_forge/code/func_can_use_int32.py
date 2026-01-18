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
def can_use_int32():
    free_symbols = list(expr.free_symbols)
    sizes = {k: v for k, v in zip(self.itervars, self.ranges) if k in free_symbols}
    if any((v == 0 for v in sizes.values())):
        return True
    vars_ranges = {k: ValueRanges(0, v - 1) for k, v in sizes.items()}
    if not vars_ranges or len(vars_ranges) != len(free_symbols):
        i32_iinfo = torch.iinfo(torch.int32)
        return expr.is_number and expr <= i32_iinfo.max and (expr >= i32_iinfo.min)
    expr_ranges = bound_sympy(expr, vars_ranges)
    if math.isinf(expr_ranges.lower) or math.isinf(expr_ranges.upper):
        return False
    return range_expressable_in_32_bits(ValueRanges(int(expr_ranges.lower), int(expr_ranges.upper) + 1))