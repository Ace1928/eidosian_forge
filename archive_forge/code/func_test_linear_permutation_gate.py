import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('n_elements,n_permuted', ((n_elements, random.randint(0, n_elements)) for n_elements in (random.randint(5, 20) for _ in range(20))))
def test_linear_permutation_gate(n_elements, n_permuted):
    qubits = cirq.LineQubit.range(n_elements)
    elements = tuple(range(n_elements))
    elements_to_permute = random.sample(elements, n_permuted)
    permuted_elements = random.sample(elements_to_permute, n_permuted)
    permutation = {e: p for e, p in zip(elements_to_permute, permuted_elements)}
    cca.PermutationGate.validate_permutation(permutation, n_elements)
    gate = cca.LinearPermutationGate(n_elements, permutation)
    ct.assert_equivalent_repr(gate)
    assert gate.permutation() == permutation
    mapping = dict(zip(qubits, elements))
    for swap in cirq.flatten_op_tree(cirq.decompose_once_with_qubits(gate, qubits)):
        assert isinstance(swap, cirq.GateOperation)
        swap.gate.update_mapping(mapping, swap.qubits)
    for i in range(n_elements):
        p = permutation.get(elements[i], i)
        assert mapping.get(qubits[p], elements[i]) == i