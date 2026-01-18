import cirq
import cirq_web
import pytest
def test_circuit_init_type():
    qubits = [cirq.GridQubit(x, y) for x in range(2) for y in range(2)]
    moment = cirq.Moment(cirq.H(qubits[0]))
    circuit = cirq.Circuit(moment)
    circuit3d = cirq_web.Circuit3D(circuit)
    assert isinstance(circuit3d, cirq_web.Circuit3D)