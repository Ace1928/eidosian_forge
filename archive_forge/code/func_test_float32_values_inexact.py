import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_float32_values_inexact(self):
    tp = np.float32
    for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
        a = scale * 1.0
        b = scale * (1.0 + FLT_EPSILON)
        c = scale * (1.0 + FLT_EPSILON * 2)
        d = scale * (1.0 + FLT_EPSILON * 4)
        self.ne(tp(a), tp(b))
        self.ne(tp(a), tp(b), prec='exact')
        self.ne(tp(a), tp(b), prec='double')
        self.eq(tp(a), tp(b), prec='single')
        self.ne(tp(a), tp(c), prec='single')
        self.eq(tp(a), tp(c), prec='single', ulps=2)
        self.ne(tp(a), tp(d), prec='single', ulps=2)
        self.eq(tp(a), tp(c), prec='single', ulps=3)
        self.eq(tp(a), tp(d), prec='single', ulps=3)
    self.eq(tp(1e-07), tp(2e-07), prec='single', abs_tol='eps')
    self.ne(tp(1e-07), tp(3e-07), prec='single', abs_tol='eps')
    self.eq(tp(1e-07), tp(1e-08), prec='single', abs_tol=1e-07)
    self.ne(tp(1e-07), tp(3e-07), prec='single', abs_tol=1e-07)