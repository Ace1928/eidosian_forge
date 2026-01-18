from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_special_numerics():
    assert not cirq.approx_eq(float('nan'), 0, atol=0.0)
    assert not cirq.approx_eq(float('nan'), float('nan'), atol=0.0)
    assert not cirq.approx_eq(float('inf'), float('-inf'), atol=0.0)
    assert not cirq.approx_eq(float('inf'), 5, atol=0.0)
    assert not cirq.approx_eq(float('inf'), 0, atol=0.0)
    assert cirq.approx_eq(float('inf'), float('inf'), atol=0.0)