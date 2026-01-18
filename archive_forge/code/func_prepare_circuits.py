from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def prepare_circuits(*, num_qubits: int, depth: int, num_circuits: int, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List[Tuple[cirq.Circuit, List[int]]]:
    """Generates circuits and computes their heavy set.

    Args:
        num_qubits: The number of qubits in the generated circuits.
        depth: The number of layers in the circuits.
        num_circuits: The number of circuits to create.
        random_state: Random state or random state seed.

    Returns:
        A list of tuples where the first element is a generated model
        circuit and the second element is the heavy set for that circuit.
    """
    circuits = []
    print('Computing heavy sets')
    for circuit_i in range(num_circuits):
        model_circuit = generate_model_circuit(num_qubits, depth, random_state=random_state)
        heavy_set = compute_heavy_set(model_circuit)
        print(f'  Circuit {circuit_i + 1} Heavy Set: {heavy_set}')
        circuits.append((model_circuit, heavy_set))
    return circuits