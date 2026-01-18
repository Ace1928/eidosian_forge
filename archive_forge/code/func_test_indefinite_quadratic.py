import warnings
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_indefinite_quadratic(self) -> None:
    x = Variable()
    y = Variable()
    z = Variable()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = y * z
        self.assertTrue(s.is_quadratic())
        self.assertFalse(s.is_dcp())
        t = (x + y) ** 2 - s - z * z
        self.assertTrue(t.is_quadratic())
        self.assertFalse(t.is_dcp())