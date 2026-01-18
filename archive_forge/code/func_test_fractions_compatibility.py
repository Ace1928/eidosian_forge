from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_fractions_compatibility():
    assert cirq.approx_eq(Fraction(0), Fraction(1, int(10000000000.0)), atol=1e-09)
    assert not cirq.approx_eq(Fraction(0), Fraction(1, int(10000000.0)), atol=1e-09)