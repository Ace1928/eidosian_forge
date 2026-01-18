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
def mark_parallel(self, par_depth):
    assert par_depth <= self.max_parallel_depth(), 'Parallel depth cannot exceed the maximal allowed parallel depth'
    assert self.root is not None
    loops = self.root
    for loop in loops:
        loop.parallel = par_depth
    for i in range(1, par_depth):
        loops = loops[0].inner
        loops[0].collapsed = True