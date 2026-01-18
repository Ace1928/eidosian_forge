import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_float64_values_inexact(self):
    for tp in [float, np.float64]:
        for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
            a = scale * 1.0
            b = scale * (1.0 + DBL_EPSILON)
            c = scale * (1.0 + DBL_EPSILON * 2)
            d = scale * (1.0 + DBL_EPSILON * 4)
            self.ne(tp(a), tp(b))
            self.ne(tp(a), tp(b), prec='exact')
            self.eq(tp(a), tp(b), prec='double')
            self.eq(tp(a), tp(b), prec='double', ulps=1)
            self.ne(tp(a), tp(c), prec='double')
            self.eq(tp(a), tp(c), prec='double', ulps=2)
            self.ne(tp(a), tp(d), prec='double', ulps=2)
            self.eq(tp(a), tp(c), prec='double', ulps=3)
            self.eq(tp(a), tp(d), prec='double', ulps=3)
        self.eq(tp(1e-16), tp(3e-16), prec='double', abs_tol='eps')
        self.ne(tp(1e-16), tp(4e-16), prec='double', abs_tol='eps')
        self.eq(tp(1e-17), tp(1e-18), prec='double', abs_tol=1e-17)
        self.ne(tp(1e-17), tp(3e-17), prec='double', abs_tol=1e-17)