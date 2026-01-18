from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_decimal_compatibility():
    assert cirq.approx_eq(Decimal('0'), Decimal('0.0000000001'), atol=1e-09)
    assert not cirq.approx_eq(Decimal('0'), Decimal('0.00000001'), atol=1e-09)
    assert not cirq.approx_eq(Decimal('NaN'), Decimal('-Infinity'), atol=1e-09)