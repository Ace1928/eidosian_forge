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
def test_looping1(self):
    rec = self.compile_and_record(looping_usecase1)
    self.assertFalse(rec.alive)
    self.assertRecordOrder(rec, ['a', 'b', '--loop exit--', 'c'])
    self.assertRecordOrder(rec, ['iter(a)#1', '--loop bottom--', 'iter(a)#2', '--loop bottom--', 'iter(a)#3', '--loop bottom--', 'iter(a)', '--loop exit--'])