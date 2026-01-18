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
def reduction_acc_type_vec(reduction_type, dtype):
    assert reduction_type not in {'argmin', 'argmax'}
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    vec_type = f'at::vec::Vectorized<{scalar_type}>'
    if is_welford_reduction(reduction_type):
        return f'Welford<{vec_type}>'
    return vec_type