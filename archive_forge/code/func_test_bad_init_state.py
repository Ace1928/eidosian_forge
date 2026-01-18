import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_bad_init_state():
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    with pytest.raises(ValueError):
        ccq.circuit_to_tensors(circuit=circuit, qubits=qubits, initial_state=1)