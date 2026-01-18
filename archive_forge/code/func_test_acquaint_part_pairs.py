from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('part_lens', [tuple((randint(1, 3) for _ in range(randint(2, 10)))) for _ in range(3)])
def test_acquaint_part_pairs(part_lens):
    parts = []
    n_qubits = 0
    for part_len in part_lens:
        parts.append(tuple(range(n_qubits, n_qubits + part_len)))
        n_qubits += part_len
    qubits = cirq.LineQubit.range(n_qubits)
    swap_network_op = cca.SwapNetworkGate(part_lens, acquaintance_size=None)(*qubits)
    swap_network = cirq.Circuit(swap_network_op)
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    actual_opps = cca.get_logical_acquaintance_opportunities(swap_network, initial_mapping)
    expected_opps = set((frozenset(s + t) for s, t in combinations(parts, 2)))
    assert expected_opps == actual_opps