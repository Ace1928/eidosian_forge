from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
def test_assert_implements_consistent_protocols():
    cirq.testing.assert_implements_consistent_protocols(GoodGate(phase_exponent=0.0), global_vals={'GoodGate': GoodGate})
    cirq.testing.assert_implements_consistent_protocols(GoodGate(phase_exponent=0.25), global_vals={'GoodGate': GoodGate})
    cirq.testing.assert_implements_consistent_protocols(GoodGate(phase_exponent=sympy.Symbol('t')), global_vals={'GoodGate': GoodGate})
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGateIsParameterized(phase_exponent=0.25))
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGateParameterNames(phase_exponent=0.25))
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGateApplyUnitaryToTensor(phase_exponent=0.25))
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGateDecompose(phase_exponent=0.25))
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGatePauliExpansion(phase_exponent=0.25))
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGatePhaseBy(phase_exponent=0.25))
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(BadGateRepr(phase_exponent=0.25), global_vals={'BadGateRepr': BadGateRepr})
    with pytest.raises(AssertionError):
        cirq.testing.assert_implements_consistent_protocols(controlled_gate_op_test.BadGate())