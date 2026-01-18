import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_get_grid_interaction_layer_circuit():
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 3))
    layer_circuit = get_grid_interaction_layer_circuit(graph)
    sqrtisw = cirq.ISWAP ** 0.5
    gq = cirq.GridQubit
    should_be = cirq.Circuit(sqrtisw(gq(0, 0), gq(1, 0)), sqrtisw(gq(1, 1), gq(2, 1)), sqrtisw(gq(0, 2), gq(1, 2)), sqrtisw(gq(0, 1), gq(1, 1)), sqrtisw(gq(1, 2), gq(2, 2)), sqrtisw(gq(1, 0), gq(2, 0)), sqrtisw(gq(0, 1), gq(0, 2)), sqrtisw(gq(1, 0), gq(1, 1)), sqrtisw(gq(2, 1), gq(2, 2)), sqrtisw(gq(0, 0), gq(0, 1)), sqrtisw(gq(1, 1), gq(1, 2)), sqrtisw(gq(2, 0), gq(2, 1)))
    assert layer_circuit == should_be