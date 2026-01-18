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
def test_looping2(self):
    rec = self.compile_and_record(looping_usecase2)
    self.assertFalse(rec.alive)
    self.assertRecordOrder(rec, ['a', '--outer loop top--'])
    self.assertRecordOrder(rec, ['iter(a)', '--outer loop else--', '--outer loop exit--'])
    self.assertRecordOrder(rec, ['iter(b)', '--inner loop exit #1--', 'iter(b)', '--inner loop exit #2--', 'iter(b)', '--inner loop exit #3--'])
    self.assertRecordOrder(rec, ['iter(a)#1', '--inner loop entry #1--', 'iter(a)#2', '--inner loop entry #2--', 'iter(a)#3', '--inner loop entry #3--'])
    self.assertRecordOrder(rec, ['iter(a)#1 + iter(a)#1', '--outer loop bottom #1--'])