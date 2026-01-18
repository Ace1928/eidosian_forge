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
def select_tiling_indices():
    all_index = []
    for node in nodes:
        rw = dependencies.extract_read_writes(node._body, *node._sizes)
        all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
    contig_vars = set()
    contig_vars_list = []
    non_contig_stride_const = set()
    non_contig_stride_other = set()
    for index in all_index:
        for var in index.free_symbols:
            if not re.search('^d\\d+$', var.name):
                continue
            stride = stride_at(var, index)
            if stride == 1:
                contig_vars.add(int(var.name[1:]))
                contig_vars_list.append(int(var.name[1:]))
            elif all((s.name.startswith('s') for s in stride.free_symbols)):
                non_contig_stride_const.add(int(var.name[1:]))
            else:
                non_contig_stride_other.add(int(var.name[1:]))
    contig_only = contig_vars - non_contig_stride_const - non_contig_stride_other
    if len(contig_vars) == 0:
        return [len(self.itervars) - 1]
    if contig_only:
        return sorted(contig_only)[-1:]
    contig_and_const_stride = (contig_vars & non_contig_stride_const) - non_contig_stride_other
    contig_vars_sorted = sorted(contig_vars)
    if len(contig_vars_sorted) == 2 and contig_vars_sorted[-1] in contig_and_const_stride and (contig_vars_sorted[-1] == len(self.itervars) - 1):
        return contig_vars_sorted
    return sorted(contig_vars_sorted, key=contig_vars_list.count)[-1:]