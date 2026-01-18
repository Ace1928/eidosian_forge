import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_generate_library_of_2q_circuits_custom_qubits():
    circuits = generate_library_of_2q_circuits(n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=13, q0=cirq.GridQubit(9, 9), q1=cirq.NamedQubit('hi mom'), random_state=9)
    assert len(circuits) == 5
    for circuit in circuits:
        assert sorted(circuit.all_qubits()) == [cirq.GridQubit(9, 9), cirq.NamedQubit('hi mom')]
        for m1, m2 in zip(circuit.moments[::2], circuit.moments[1::2]):
            assert len(m1.operations) == 2
            assert len(m2.operations) == 1
            assert m2.operations[0].gate == cirq.ISWAP ** 0.5