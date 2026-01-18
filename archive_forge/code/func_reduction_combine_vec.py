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
def reduction_combine_vec(reduction_type, var, next_value):
    if reduction_type == 'max':
        return f'at::vec::maximum({var}, {next_value})'
    elif reduction_type == 'min':
        return f'at::vec::minimum({var}, {next_value})'
    elif reduction_type == 'sum':
        return f'{var} + {next_value}'
    elif reduction_type == 'prod':
        return f'{var} * {next_value}'
    elif reduction_type == 'xor_sum':
        return f'{var} ^ {next_value}'
    elif reduction_type == 'welford_reduce':
        return f'welford_combine({var}, {next_value})'
    elif reduction_type == 'welford_combine':
        if isinstance(next_value, tuple):
            mean, m2, weight = next_value
        else:
            mean, m2, weight = reduction_project(reduction_type, next_value)
        return f'welford_combine({var}, {{{mean}, {m2}, {weight}}})'
    else:
        raise NotImplementedError()