import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def is_del_grouped_at_the_end(fir):
    [blk] = fir.blocks.values()
    inst_is_del = [isinstance(stmt, ir.Del) for stmt in blk.body]
    not_dels = list(takewhile(operator.not_, inst_is_del))
    begin = len(not_dels)
    all_dels = list(takewhile(operator.truth, inst_is_del[begin:]))
    end = begin + len(all_dels)
    return end == len(inst_is_del) - 1