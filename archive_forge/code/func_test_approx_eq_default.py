from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_default():
    assert cirq.approx_eq(1.0, 1.0 + 1e-09)
    assert cirq.approx_eq(1.0, 1.0 - 1e-09)
    assert not cirq.approx_eq(1.0, 1.0 + 1e-07)
    assert not cirq.approx_eq(1.0, 1.0 - 1e-07)