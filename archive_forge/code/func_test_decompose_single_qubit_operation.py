import pytest
import sympy
import cirq
from cirq_aqt import aqt_target_gateset
def test_decompose_single_qubit_operation():
    gs = aqt_target_gateset.AQTTargetGateset()
    tgoph = gs.decompose_to_target_gateset(cirq.H(Q), 0)
    assert len(tgoph) == 2
    assert isinstance(tgoph[0].gate, cirq.Rx)
    assert isinstance(tgoph[1].gate, cirq.Ry)
    tcoph = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.H(Q))).with_tags('tagged')
    tgtcoph = gs.decompose_to_target_gateset(tcoph, 0)
    assert len(tgtcoph) == 2
    assert isinstance(tgtcoph[0].gate, cirq.Rx)
    assert isinstance(tgtcoph[1].gate, cirq.Ry)
    tgopz = gs.decompose_to_target_gateset(cirq.Z(Q), 0)
    assert len(tgopz) == 1
    assert isinstance(tgopz[0].gate, cirq.ZPowGate)
    theta = sympy.Symbol('theta')
    assert gs.decompose_to_target_gateset(cirq.H(Q) ** theta, 0) is NotImplemented
    return