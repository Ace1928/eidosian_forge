import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import (
from numba.misc.special import literally
def test_mutual_recursion_literal(self):

    def get_functions(decor):

        @decor
        def outer_fac(n, value):
            if n < 1:
                return value
            return n * inner_fac(n - 1, value)

        @decor
        def inner_fac(n, value):
            if n < 1:
                return literally(value)
            return n * outer_fac(n - 1, value)
        return (outer_fac, inner_fac)
    ref_outer_fac, ref_inner_fac = get_functions(lambda x: x)
    outer_fac, inner_fac = get_functions(njit)
    self.assertEqual(outer_fac(10, 12), ref_outer_fac(10, 12))
    self.assertEqual(outer_fac.signatures[0][1].literal_value, 12)
    self.assertEqual(inner_fac.signatures[0][1].literal_value, 12)
    self.assertEqual(inner_fac(11, 13), ref_inner_fac(11, 13))
    self.assertEqual(outer_fac.signatures[1][1].literal_value, 13)
    self.assertEqual(inner_fac.signatures[1][1].literal_value, 13)