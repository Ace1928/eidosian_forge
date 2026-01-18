import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_complex128_values_inexact(self):
    for tp in [complex, np.complex128]:
        for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
            a = scale * 1.0
            b = scale * (1.0 + DBL_EPSILON)
            c = scale * (1.0 + DBL_EPSILON * 2)
            aa = tp(complex(a, a))
            ab = tp(complex(a, b))
            bb = tp(complex(b, b))
            self.ne(tp(aa), tp(ab))
            self.eq(tp(aa), tp(ab), prec='double')
            self.eq(tp(ab), tp(bb), prec='double')
            self.eq(tp(aa), tp(bb), prec='double')
            ac = tp(complex(a, c))
            cc = tp(complex(c, c))
            self.ne(tp(aa), tp(ac), prec='double')
            self.ne(tp(ac), tp(cc), prec='double')
            self.eq(tp(aa), tp(ac), prec='double', ulps=2)
            self.eq(tp(ac), tp(cc), prec='double', ulps=2)
            self.eq(tp(aa), tp(cc), prec='double', ulps=2)
            self.eq(tp(aa), tp(cc), prec='single')