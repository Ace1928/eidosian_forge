import random
import numpy as np
import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def test_initialization_with_no_seed():
    graph_seed = random.randint(0, 2 ** 32)
    state = np.random.get_state()
    mappings = []
    for _ in range(3):
        np.random.set_state(state)
        mappings.append(get_seeded_initial_mapping(graph_seed, None))
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*mappings)