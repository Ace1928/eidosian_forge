import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
def test_iterable_type_constructor(self):

    @njit
    def func1(a, b):
        d = Dict(zip(a, b))
        return d

    @njit
    def func2(a_, b):
        a = range(3)
        return Dict(zip(a, b))

    @njit
    def func3(a_, b):
        a = [0, 1, 2]
        return Dict(zip(a, b))

    @njit
    def func4(a, b):
        c = zip(a, b)
        return Dict(zip(a, zip(c, a)))

    @njit
    def func5(a, b):
        return Dict(zip(zip(a, b), b))

    @njit
    def func6(items):
        return Dict(items)

    @njit
    def func7(k, v):
        return Dict({k: v})

    @njit
    def func8(k, v):
        d = Dict()
        d[k] = v
        return d

    def _get_dict(py_dict):
        d = Dict()
        for k, v in py_dict.items():
            d[k] = v
        return d
    vals = ((func1, [(0, 1, 2), 'abc'], _get_dict({0: 'a', 1: 'b', 2: 'c'})), (func2, [(0, 1, 2), 'abc'], _get_dict({0: 'a', 1: 'b', 2: 'c'})), (func3, [(0, 1, 2), 'abc'], _get_dict({0: 'a', 1: 'b', 2: 'c'})), (func4, [(0, 1, 2), 'abc'], _get_dict({0: ((0, 'a'), 0), 1: ((1, 'b'), 1), 2: ((2, 'c'), 2)})), (func5, [(0, 1, 2), 'abc'], _get_dict({(0, 'a'): 'a', (1, 'b'): 'b', (2, 'c'): 'c'})), (func6, [((1, 'a'), (3, 'b'))], _get_dict({1: 'a', 3: 'b'})), (func1, ['key', _get_dict({1: 'abc'})], _get_dict({'k': 1})), (func8, ['key', _get_dict({1: 'abc'})], _get_dict({'key': _get_dict({1: 'abc'})})), (func8, ['key', List([1, 2, 3])], _get_dict({'key': List([1, 2, 3])})))
    for func, args, expected in vals:
        if self.jit_enabled:
            got = func(*args)
        else:
            got = func.py_func(*args)
        self.assertPreciseEqual(expected, got)