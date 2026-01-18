import pytest
import sympy
import cirq
from cirq_aqt import aqt_target_gateset
def test_decompose_two_qubit_operation():
    gs = aqt_target_gateset.AQTTargetGateset()
    tgopsqrtxx = gs.decompose_to_target_gateset(cirq.XX(Q, Q2) ** 0.5, 0)
    assert len(tgopsqrtxx) == 1
    assert isinstance(tgopsqrtxx[0].gate, cirq.XXPowGate)
    theta = sympy.Symbol('theta')
    assert gs.decompose_to_target_gateset(cirq.XX(Q, Q2) ** theta, 0) is NotImplemented
    return