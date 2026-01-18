import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_get_random_combinations_for_small_device():
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 1))
    n_combinations = 4
    combinations = get_random_combinations_for_device(n_library_circuits=3, n_combinations=n_combinations, device_graph=graph, random_state=99)
    assert len(combinations) == 2