import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('num_qubits,permutation,inverse', [(2, {0: 1, 1: 0}, {0: 1, 1: 0}), (3, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}), (3, {0: 1, 1: 2, 2: 0}, {0: 2, 1: 0, 2: 1}), (3, {0: 2, 1: 0, 2: 1}, {0: 1, 1: 2, 2: 0}), (4, {0: 3, 1: 2, 2: 1, 3: 0}, {0: 3, 1: 2, 2: 1, 3: 0})])
def test_linear_permutation_gate_pow_inverse(num_qubits, permutation, inverse):
    permutation_gate = cca.LinearPermutationGate(num_qubits, permutation)
    inverse_gate = cca.LinearPermutationGate(num_qubits, inverse)
    assert permutation_gate ** (-1) == inverse_gate
    assert cirq.inverse(permutation_gate) == inverse_gate