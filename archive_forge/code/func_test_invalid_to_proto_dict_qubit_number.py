import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_invalid_to_proto_dict_qubit_number():
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        _ = programs.gate_to_proto(cirq.CZ ** 0.5, (cirq.GridQubit(2, 3),), delay=0)
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        programs.gate_to_proto(cirq.Z ** 0.5, (cirq.GridQubit(2, 3), cirq.GridQubit(3, 4)), delay=0)
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        programs.gate_to_proto(cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0), (cirq.GridQubit(2, 3), cirq.GridQubit(3, 4)), delay=0)