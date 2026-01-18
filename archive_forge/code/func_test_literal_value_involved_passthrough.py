from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def test_literal_value_involved_passthrough(self):

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):
        self.assertTrue(isinstance(x, types.LiteralStrKeyDict))
        dlv = x.literal_value
        inner_literal = {types.literal('g'): types.literal('h'), types.literal('i'): types.Array(types.float64, 1, 'C')}
        inner_dict = types.LiteralStrKeyDict(inner_literal)
        outer_literal = {types.literal('a'): types.LiteralList([types.literal(1), types.literal('a'), types.DictType(types.unicode_type, types.intp, initial_value={'f': 1}), inner_dict]), types.literal('b'): types.literal(2), types.literal('c'): types.List(types.complex128, reflected=False)}

        def check_same(a, b):
            if isinstance(a, types.LiteralList) and isinstance(b, types.LiteralList):
                for i, j in zip(a.literal_value, b.literal_value):
                    check_same(a.literal_value, b.literal_value)
            elif isinstance(a, list) and isinstance(b, list):
                for i, j in zip(a, b):
                    check_same(i, j)
            elif isinstance(a, types.LiteralStrKeyDict) and isinstance(b, types.LiteralStrKeyDict):
                for (ki, vi), (kj, vj) in zip(a.literal_value.items(), b.literal_value.items()):
                    check_same(ki, kj)
                    check_same(vi, vj)
            elif isinstance(a, dict) and isinstance(b, dict):
                for (ki, vi), (kj, vj) in zip(a.items(), b.items()):
                    check_same(ki, kj)
                    check_same(vi, vj)
            else:
                self.assertEqual(a, b)
        check_same(dlv, outer_literal)
        return lambda x: x

    @njit
    def foo():
        l = {'a': [1, 'a', {'f': 1}, {'g': 'h', 'i': np.zeros(5)}], 'b': 2, 'c': [1j, 2j, 3j]}
        bar(l)
    foo()