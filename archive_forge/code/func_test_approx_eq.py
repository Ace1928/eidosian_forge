from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
def test_approx_eq():
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(1.5), atol=0.1)
    assert cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(1.7), atol=0.3)
    assert not cirq.approx_eq(CExpZinGate(1.5), CExpZinGate(1.7), atol=0.1)
    assert cirq.approx_eq(ZGateDef(exponent=1.5), ZGateDef(exponent=1.5), atol=0.1)
    assert not cirq.approx_eq(CExpZinGate(1.5), ZGateDef(exponent=1.5), atol=0.1)
    with pytest.raises(TypeError, match=re.escape("unsupported operand type(s) for -: 'Symbol' and 'PeriodicValue'")):
        cirq.approx_eq(ZGateDef(exponent=1.5), ZGateDef(exponent=sympy.Symbol('a')), atol=0.1)
    assert cirq.approx_eq(CExpZinGate(sympy.Symbol('a')), CExpZinGate(sympy.Symbol('a')), atol=0.1)
    with pytest.raises(AttributeError, match='Insufficient information to decide whether expressions are approximately equal .* vs .*'):
        assert not cirq.approx_eq(CExpZinGate(sympy.Symbol('a')), CExpZinGate(sympy.Symbol('b')), atol=0.1)