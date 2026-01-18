import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_complex64_values_inexact(self):
    tp = np.complex64
    for scale in [1.0, -2 ** 3, 2 ** (-4), -2 ** (-20)]:
        a = scale * 1.0
        b = scale * (1.0 + FLT_EPSILON)
        c = scale * (1.0 + FLT_EPSILON * 2)
        aa = tp(complex(a, a))
        ab = tp(complex(a, b))
        bb = tp(complex(b, b))
        self.ne(tp(aa), tp(ab))
        self.ne(tp(aa), tp(ab), prec='double')
        self.eq(tp(aa), tp(ab), prec='single')
        self.eq(tp(ab), tp(bb), prec='single')
        self.eq(tp(aa), tp(bb), prec='single')
        ac = tp(complex(a, c))
        cc = tp(complex(c, c))
        self.ne(tp(aa), tp(ac), prec='single')
        self.ne(tp(ac), tp(cc), prec='single')
        self.eq(tp(aa), tp(ac), prec='single', ulps=2)
        self.eq(tp(ac), tp(cc), prec='single', ulps=2)
        self.eq(tp(aa), tp(cc), prec='single', ulps=2)