import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
def test_base_cases(self):

    def test_0():
        a = np.zeros(0)
        b = np.zeros(1)
        m = 0
        n = 1
        c = np.zeros((m, n))
        return
    self._compile_and_test(test_0, (), equivs=[self.with_equiv('a', (0,)), self.with_equiv('b', (1,)), self.with_equiv('c', (0, 1))])

    def test_1(n):
        a = np.zeros(n)
        b = np.zeros(n)
        return a + b
    self._compile_and_test(test_1, (types.intp,), asserts=None)

    def test_2(m, n):
        a = np.zeros(n)
        b = np.zeros(m)
        return a + b
    self._compile_and_test(test_2, (types.intp, types.intp), asserts=[self.with_assert('a', 'b')])

    def test_3(n):
        a = np.zeros(n)
        return a + n
    self._compile_and_test(test_3, (types.intp,), asserts=None)

    def test_4(n):
        a = np.zeros(n)
        b = a + 1
        c = a + 2
        return a + c
    self._compile_and_test(test_4, (types.intp,), asserts=None)

    def test_5(n):
        a = np.zeros((n, n))
        m = n
        b = np.zeros((m, n))
        return a + b
    self._compile_and_test(test_5, (types.intp,), asserts=None)

    def test_6(m, n):
        a = np.zeros(n)
        b = np.zeros(m)
        d = a + b
        e = a - b
        return d + e
    self._compile_and_test(test_6, (types.intp, types.intp), asserts=[self.with_assert('a', 'b'), self.without_assert('d', 'e')])

    def test_7(m, n):
        a = np.zeros(n)
        b = np.zeros(m)
        if m == 10:
            d = a + b
        else:
            d = a - b
        return d + a
    self._compile_and_test(test_7, (types.intp, types.intp), asserts=[self.with_assert('a', 'b'), self.without_assert('d', 'a')])

    def test_8(m, n):
        a = np.zeros(n)
        b = np.zeros(m)
        if m == 10:
            d = b + a
        else:
            d = a + a
        return b + d
    self._compile_and_test(test_8, (types.intp, types.intp), asserts=[self.with_assert('b', 'a'), self.with_assert('b', 'd')])

    def test_9(m):
        A = np.ones(m)
        s = 0
        while m < 2:
            m += 1
            B = np.ones(m)
            s += np.sum(A + B)
        return s
    self._compile_and_test(test_9, (types.intp,), asserts=[self.with_assert('A', 'B')])

    def test_10(m, n):
        p = m - 1
        q = n + 1
        r = q + 1
        A = np.zeros(p)
        B = np.zeros(q)
        C = np.zeros(r)
        D = np.zeros(m)
        s = np.sum(A + B)
        t = np.sum(C + D)
        return s + t
    self._compile_and_test(test_10, (types.intp, types.intp), asserts=[self.with_assert('A', 'B'), self.without_assert('C', 'D')])

    def test_11():
        a = np.ones(5)
        b = np.ones(5)
        c = a[1:]
        d = b[:-1]
        e = len(c)
        f = len(d)
        return e == f
    self._compile_and_test(test_11, (), equivs=[self.with_equiv('e', 'f')])

    def test_12():
        a = np.ones(25).reshape((5, 5))
        b = np.ones(25).reshape((5, 5))
        c = a[1:, :]
        d = b[:-1, :]
        e = c.shape[0]
        f = d.shape[0]
        g = len(d)
        return e == f
    self._compile_and_test(test_12, (), equivs=[self.with_equiv('e', 'f', 'g')])

    def test_tup_arg(T):
        T2 = T
        return T2[0]
    int_arr_typ = types.Array(types.intp, 1, 'C')
    self._compile_and_test(test_tup_arg, (types.Tuple((int_arr_typ, int_arr_typ)),), asserts=None)

    def test_arr_in_tup(m):
        A = np.ones(m)
        S = (A,)
        B = np.ones(len(S[0]))
        return B
    self._compile_and_test(test_arr_in_tup, (types.intp,), equivs=[self.with_equiv('A', 'B')])
    T = namedtuple('T', ['a', 'b'])

    def test_namedtuple(n):
        r = T(n, n)
        return r[0]
    self._compile_and_test(test_namedtuple, (types.intp,), equivs=[self.with_equiv('r', ('n', 'n'))])

    def test_np_where_tup_return(A):
        c = np.where(A)
        return len(c[0])
    self._compile_and_test(test_np_where_tup_return, (types.Array(types.intp, 1, 'C'),), asserts=None)

    def test_shape(A):
        m, n = A.shape
        B = np.ones((m, n))
        return A + B
    self._compile_and_test(test_shape, (types.Array(types.intp, 2, 'C'),), asserts=None)

    def test_cond(l, m, n):
        A = np.ones(l)
        B = np.ones(m)
        C = np.ones(n)
        if l == m:
            r = np.sum(A + B)
        else:
            r = 0
        if m != n:
            s = 0
        else:
            s = np.sum(B + C)
        t = 0
        if l == m:
            if m == n:
                t = np.sum(A + B + C)
        return r + s + t
    self._compile_and_test(test_cond, (types.intp, types.intp, types.intp), asserts=None)

    def test_assert_1(m, n):
        assert m == n
        A = np.ones(m)
        B = np.ones(n)
        return np.sum(A + B)
    self._compile_and_test(test_assert_1, (types.intp, types.intp), asserts=None)

    def test_assert_2(A, B):
        assert A.shape == B.shape
        return np.sum(A + B)
    self._compile_and_test(test_assert_2, (types.Array(types.intp, 1, 'C'), types.Array(types.intp, 1, 'C')), asserts=None)
    self._compile_and_test(test_assert_2, (types.Array(types.intp, 2, 'C'), types.Array(types.intp, 2, 'C')), asserts=None)
    with self.assertRaises(AssertionError) as raises:
        self._compile_and_test(test_assert_2, (types.Array(types.intp, 1, 'C'), types.Array(types.intp, 2, 'C')), asserts=None)
    msg = 'Dimension mismatch'
    self.assertIn(msg, str(raises.exception))