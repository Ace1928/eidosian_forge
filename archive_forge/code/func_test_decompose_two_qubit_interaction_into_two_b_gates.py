import itertools
import random
from typing import Any
import numpy as np
import pytest
import sympy
import cirq
from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
@pytest.mark.parametrize('obj', UNITARY_OBJS)
def test_decompose_two_qubit_interaction_into_two_b_gates(obj: Any):
    circuit = cirq.Circuit(_decompose_two_qubit_interaction_into_two_b_gates(obj, qubits=cirq.LineQubit.range(2)))
    desired_unitary = obj if isinstance(obj, np.ndarray) else cirq.unitary(obj)
    for operation in circuit.all_operations():
        assert len(operation.qubits) < 2 or operation.gate == _B
    np.testing.assert_allclose(cirq.unitary(circuit), desired_unitary, atol=1e-06)