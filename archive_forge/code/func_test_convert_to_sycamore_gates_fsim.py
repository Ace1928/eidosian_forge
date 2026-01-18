import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_convert_to_sycamore_gates_fsim():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1))
    compiled_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    cirq.testing.assert_same_circuits(circuit, compiled_circuit)