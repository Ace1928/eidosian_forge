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
def reduction_init(reduction_type, dtype):
    if dtype in DTYPE_LOWP_FP:
        dtype = torch.float32
    if reduction_type in ('xor_sum', 'sum', 'any'):
        return 0
    if reduction_type == 'prod':
        return 1
    if reduction_type in {'max', 'argmax'}:
        return f'-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()' if is_float_dtype(dtype) else f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()'
    if reduction_type in {'min', 'argmin'}:
        return f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()' if is_float_dtype(dtype) else f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()'
    if is_welford_reduction(reduction_type):
        return f'Welford<{DTYPE_TO_CPP[dtype]}>()'
    raise AssertionError(reduction_type)