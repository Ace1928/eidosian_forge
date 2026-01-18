from itertools import combinations
from string import ascii_lowercase
from typing import Sequence, Dict, Tuple
import numpy as np
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('num_qubits, acquaintance_size, gates', [(num_qubits, acquaintance_size, random_diagonal_gates(num_qubits, acquaintance_size)) for acquaintance_size, num_qubits in [(2, n) for n in range(2, 9)] + [(3, n) for n in range(3, 8)] + [(4, 4), (4, 6), (5, 5)] for _ in range(2)])
def test_executor_random(num_qubits: int, acquaintance_size: int, gates: Dict[Tuple[cirq.Qid, ...], cirq.Gate]):
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cca.complete_acquaintance_strategy(qubits, acquaintance_size)
    logical_circuit = cirq.Circuit([g(*Q) for Q, g in gates.items()])
    expected_unitary = logical_circuit.unitary()
    initial_mapping = {q: q for q in qubits}
    with pytest.raises(ValueError):
        cca.GreedyExecutionStrategy(gates, initial_mapping)()
    final_mapping = cca.GreedyExecutionStrategy(gates, initial_mapping)(circuit)
    permutation = {q.x: qq.x for q, qq in final_mapping.items()}
    circuit.append(cca.LinearPermutationGate(num_qubits, permutation)(*qubits))
    actual_unitary = circuit.unitary()
    np.testing.assert_allclose(actual=actual_unitary, desired=expected_unitary, verbose=True)