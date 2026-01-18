from fractions import Fraction
from decimal import Decimal
from numbers import Number
import numpy as np
import pytest
import sympy
import cirq
def test_approx_eq_primitives():
    assert not cirq.approx_eq(1, 2, atol=0.1)
    assert cirq.approx_eq(1.0, 1.0 + 1e-10, atol=1e-09)
    assert not cirq.approx_eq(1.0, 1.0 + 1e-10, atol=1e-11)
    assert cirq.approx_eq(0.0, 1e-10, atol=1e-09)
    assert not cirq.approx_eq(0.0, 1e-10, atol=1e-11)
    assert cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), atol=0.3)
    assert not cirq.approx_eq(complex(1, 1), complex(1.1, 1.2), atol=0.1)
    assert cirq.approx_eq('ab', 'ab', atol=0.001)
    assert not cirq.approx_eq('ab', 'ac', atol=0.001)
    assert not cirq.approx_eq('1', '2', atol=999)
    assert not cirq.approx_eq('test', 1, atol=0.001)
    assert not cirq.approx_eq('1', 1, atol=0.001)