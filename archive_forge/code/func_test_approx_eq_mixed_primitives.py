from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_mixed_primitives():
    assert cirq.approx_eq(complex(1, 1e-10), 1, atol=1e-09)
    assert not cirq.approx_eq(complex(1, 0.0001), 1, atol=1e-09)
    assert cirq.approx_eq(complex(1, 1e-10), 1.0, atol=1e-09)
    assert not cirq.approx_eq(complex(1, 1e-08), 1.0, atol=1e-09)
    assert cirq.approx_eq(1, 1.0 + 1e-10, atol=1e-09)
    assert not cirq.approx_eq(1, 1.0 + 1e-10, atol=1e-11)