import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.acquaintance.strategies.cubic as ccasc
@pytest.mark.parametrize('n_qubits', range(3, 9))
def test_cubic_acquaintance_strategy(n_qubits):
    qubits = tuple(cirq.LineQubit.range(n_qubits))
    strategy = cca.cubic_acquaintance_strategy(qubits)
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    opps = cca.get_logical_acquaintance_opportunities(strategy, initial_mapping)
    assert set((len(opp) for opp in opps)) == set([3])
    expected_opps = set((frozenset(ijk) for ijk in itertools.combinations(range(n_qubits), 3)))
    assert opps == expected_opps