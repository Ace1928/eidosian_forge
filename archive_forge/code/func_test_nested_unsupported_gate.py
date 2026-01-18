import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_nested_unsupported_gate():

    class UnknownGate(cirq.testing.TwoQubitGate):
        pass
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    subcircuit = cirq.FrozenCircuit(UnknownGate()(q0, q1))
    circuit = cirq.Circuit(cirq.CircuitOperation(subcircuit))
    with pytest.raises(ValueError, match='Unable to convert'):
        cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset(), ignore_failures=False)