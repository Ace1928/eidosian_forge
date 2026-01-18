import unittest
from cvxpy import Constant, Variable, log
from cvxpy.settings import QUASILINEAR, UNKNOWN
def test_is_curvature(self) -> None:
    assert self.const.is_affine()
    assert self.aff.is_affine()
    assert not self.cvx.is_affine()
    assert not self.ccv.is_affine()
    assert not self.unknown_curv.is_affine()
    assert self.const.is_convex()
    assert self.aff.is_convex()
    assert self.cvx.is_convex()
    assert not self.ccv.is_convex()
    assert not self.unknown_curv.is_convex()
    assert self.const.is_concave()
    assert self.aff.is_concave()
    assert not self.cvx.is_concave()
    assert self.ccv.is_concave()
    assert not self.unknown_curv.is_concave()