from typing import Union
import numpy as np
from typing import List, Optional, cast
from pyquil.external.rpcq import (
import networkx as nx
def graph_to_compiler_isa(graph: nx.Graph, gates_1q: Optional[List[str]]=None, gates_2q: Optional[List[str]]=None) -> CompilerISA:
    """
    Generate an ``CompilerISA`` object from a NetworkX graph and list of 1Q and 2Q gates.
    May raise ``GraphGateError`` if the specified gates are not supported.

    :param graph: The graph topology of the quantum_processor.
    :param gates_1q: A list of 1Q gate names to be made available for all qubits in the quantum_processor.
           Defaults to ``DEFAULT_1Q_GATES``.
    :param gates_2q: A list of 2Q gate names to be made available for all edges in the quantum_processor.
           Defaults to ``DEFAULT_2Q_GATES``.
    """
    gates_1q = gates_1q or DEFAULT_1Q_GATES.copy()
    gates_2q = gates_2q or DEFAULT_2Q_GATES.copy()
    quantum_processor = CompilerISA()
    qubit_gates = []
    for gate in gates_1q:
        qubit_gates.extend(_transform_qubit_operation_to_gates(gate))
    all_qubits = list(range(max(graph.nodes) + 1))
    for i in all_qubits:
        qubit = add_qubit(quantum_processor, i)
        qubit.gates = qubit_gates
        qubit.dead = i not in graph.nodes
    edge_gates = []
    for gate in gates_2q:
        edge_gates.extend(_transform_edge_operation_to_gates(gate))
    for a, b in graph.edges:
        edge = add_edge(quantum_processor, a, b)
        edge.gates = edge_gates
    return quantum_processor