import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('permutation_sets', [random_permutation_equality_groups(5, 3, 10, 0.5)])
def test_linear_permutation_gate_equality(permutation_sets):
    swap_gates = [cirq.SWAP, cirq.CNOT]
    equals_tester = ct.EqualsTester()
    for swap_gate in swap_gates:
        for permutation_set in permutation_sets:
            equals_tester.add_equality_group(*(cca.LinearPermutationGate(10, permutation, swap_gate) for permutation in permutation_set))