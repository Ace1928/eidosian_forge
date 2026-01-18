from symengine import (
from symengine.test_utilities import raises
import unittest
def test_dirichlet_eta():
    assert dirichlet_eta(0) == Rational(1, 2)
    assert dirichlet_eta(-1) == Rational(1, 4)
    assert dirichlet_eta(1) == log(2)
    assert dirichlet_eta(2) == pi ** 2 / 12
    assert dirichlet_eta(4) == pi ** 4 * Rational(7, 720)