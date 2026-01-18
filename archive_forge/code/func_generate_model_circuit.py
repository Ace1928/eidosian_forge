from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def generate_model_circuit(num_qubits: int, depth: int, *, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> cirq.Circuit:
    """Generates a model circuit with the given number of qubits and depth.

    The generated circuit consists of `depth` layers of random qubit
    permutations followed by random two-qubit gates that are sampled from the
    Haar measure on SU(4).

    Args:
        num_qubits: The number of qubits in the generated circuit.
        depth: The number of layers in the circuit.
        random_state: Random state or random state seed.

    Returns:
        The generated circuit.
    """
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    random_state = cirq.value.parse_random_state(random_state)
    for _ in range(depth):
        perm = random_state.permutation(num_qubits)
        for k in range(0, num_qubits - 1, 2):
            permuted_indices = [int(perm[k]), int(perm[k + 1])]
            special_unitary = cirq.testing.random_special_unitary(4, random_state=random_state)
            circuit.append(cirq.MatrixGate(special_unitary).on(qubits[permuted_indices[0]], qubits[permuted_indices[1]]))
    return circuit