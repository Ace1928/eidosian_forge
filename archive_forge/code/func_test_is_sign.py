from cvxpy import Constant, Variable
from cvxpy.tests.base_test import BaseTest
def test_is_sign(self) -> None:
    assert self.pos.is_nonneg()
    assert not self.neg.is_nonneg()
    assert not self.unknown.is_nonneg()
    assert self.zero.is_nonneg()
    assert not self.pos.is_nonpos()
    assert self.neg.is_nonpos()
    assert not self.unknown.is_nonpos()
    assert self.zero.is_nonpos()
    assert self.zero.is_zero()
    assert not self.neg.is_zero()
    assert not (self.unknown.is_nonneg() or self.unknown.is_nonpos())