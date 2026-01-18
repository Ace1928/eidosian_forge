import itertools
from typing import Iterable, Tuple, Dict
import networkx as nx
import cirq
def gridqubits_to_graph_device(qubits: Iterable[cirq.GridQubit]):
    """Gets the graph of a set of grid qubits."""
    return nx.Graph((pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1))