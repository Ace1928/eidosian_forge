import numpy as np
import cirq
import pytest
def test_equality_of_gates():
    state = np.array([1, 0, 0, 0], dtype=np.complex64)
    gate_1 = cirq.StatePreparationChannel(state)
    gate_2 = cirq.StatePreparationChannel(state)
    assert gate_1 == gate_2, 'Equal state not leading to same gate'
    assert not gate_1 == state, "Incompatible objects shouldn't be equal"
    state = np.array([0, 1, 0, 0], dtype=np.complex64)
    gate_3 = cirq.StatePreparationChannel(state, name='gate_a')
    gate_4 = cirq.StatePreparationChannel(state, name='gate_b')
    assert gate_3 == gate_4, 'Equal state with different names not leading to same gate'
    assert gate_1 != gate_3, "Different states shouldn't lead to same gate"