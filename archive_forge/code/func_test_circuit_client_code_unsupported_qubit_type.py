import cirq
import cirq_web
import pytest
def test_circuit_client_code_unsupported_qubit_type():
    moment = cirq.Moment(cirq.H(cirq.NamedQubit('q0')))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        circuit.get_client_code()