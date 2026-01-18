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
def decide_parallel_depth(self, ranges, threads):
    seq = self.size_hint()
    par = 1
    depth = 0
    for expr in ranges:
        hint = V.graph.sizevars.size_hint(expr, fallback=8192)
        if par >= 2 * threads or par == threads:
            break
        if seq // threads < config.cpp.min_chunk_size:
            break
        depth += 1
        par *= hint
        seq /= hint
    if config.cpp.dynamic_threads and depth == 0 and (len(ranges) > 0):
        depth = 1
    return depth