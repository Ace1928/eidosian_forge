import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_get_random_combinations_for_pairs():
    all_pairs = [[(cirq.LineQubit(0), cirq.LineQubit(1)), (cirq.LineQubit(2), cirq.LineQubit(3))], [(cirq.LineQubit(1), cirq.LineQubit(2))]]
    combinations = get_random_combinations_for_pairs(n_library_circuits=3, n_combinations=4, all_pairs=all_pairs, random_state=99)
    assert len(combinations) == len(all_pairs)
    for i, comb in enumerate(combinations):
        assert comb.combinations.shape[0] == 4
        assert comb.combinations.shape[1] == len(comb.pairs)
        assert np.all(comb.combinations >= 0)
        assert np.all(comb.combinations < 3)
        for q0, q1 in comb.pairs:
            assert q0 in cirq.LineQubit.range(4)
            assert q1 in cirq.LineQubit.range(4)
        assert comb.layer is None
        assert comb.pairs == all_pairs[i]