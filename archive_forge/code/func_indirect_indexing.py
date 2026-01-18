import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
@staticmethod
def indirect_indexing(var, size, check=True):
    if var.bounds.lower < 0:
        new_bounds = ValueRanges.unknown()
        if var.bounds != ValueRanges.unknown() and isinstance(size, sympy.Number):
            neg = var.bounds & ValueRanges(-sympy.oo, -1)
            new_bounds = ValueRanges(neg.lower + size, neg.upper + size)
            if var.bounds.upper >= 0:
                pos = var.bounds & ValueRanges(0, sympy.oo)
                new_bounds = new_bounds | pos
        stm = ops.add(var, self.rename_indexing(size))
        if var.bounds.upper >= 0:
            lt = ops.lt(var, '0')
            stm = ops.where(lt, stm, var)
        new_var = self.cse.generate(self.compute, stm, bounds=new_bounds)
        new_var.update_on_args('index_wrap', (var,), {})
        var = new_var
    if self.generate_assert(check):
        mask = self.load_mask(var)
        map_key = (var, mask)
        existing_size, _ = self.indirect_max_sizes.get(map_key, (None, None))
        if existing_size is not None:
            size = sympy.Min(size, existing_size)
        else:
            line = '{assert_fn}({cond}, "index out of bounds: {cond_print}")'
            self.compute.writeline(IndirectAssertLine(line, self.assert_function, var, mask, self.indirect_max_sizes))
        self.indirect_max_sizes[map_key] = (size, self.index_to_str(size))
    return sympy_symbol(str(var))