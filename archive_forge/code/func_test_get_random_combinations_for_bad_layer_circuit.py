import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_get_random_combinations_for_bad_layer_circuit():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.H.on_each(q0, q1, q2, q3), cirq.CNOT(q0, q1), cirq.CNOT(q2, q3), cirq.CNOT(q1, q2))
    with pytest.raises(ValueError, match='non-2-qubit operation'):
        _ = get_random_combinations_for_layer_circuit(n_library_circuits=3, n_combinations=4, layer_circuit=circuit, random_state=99)