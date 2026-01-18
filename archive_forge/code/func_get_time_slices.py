import operator
from typing import Callable, Iterable, List, TYPE_CHECKING
import re
import networkx as nx
from cirq import circuits, ops
import cirq.contrib.acquaintance as cca
from cirq.contrib.circuitdag import CircuitDag
from cirq.contrib.routing.swap_network import SwapNetwork
def get_time_slices(dag: CircuitDag) -> List[nx.Graph]:
    """Slices the DAG into logical graphs.

    Each time slice is a graph whose vertices are qubits and whose edges
    correspond to two-qubit gates. Single-qubit gates are ignored (and
    more-than-two-qubit gates are not supported).

    The edges of the first time slice correspond to the nodes of the DAG without
    predecessors. (Again, single-qubit gates are ignored.) The edges of the
    second slice correspond to the nodes of the DAG whose only predecessors are
    in the first time slice, and so on.
    """
    circuit = circuits.Circuit((op for op in dag.all_operations() if len(op.qubits) > 1))
    return [nx.Graph((op.qubits for op in moment.operations)) for moment in circuit]