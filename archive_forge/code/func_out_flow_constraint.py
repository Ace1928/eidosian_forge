import itertools
from typing import (
import networkx as nx
import rustworkx as rx
import numpy as np
import pennylane as qml
from pennylane.ops import Hamiltonian
def out_flow_constraint(graph: Union[nx.DiGraph, rx.PyDiGraph]) -> Hamiltonian:
    """Calculates the `out flow constraint <https://1qbit.com/whitepaper/arbitrage/>`__
    Hamiltonian for the maximum-weighted cycle problem.

    Given a subset of edges in a directed graph, the out-flow constraint imposes that at most one
    edge can leave any given node, i.e., for all :math:`i`:

    .. math:: \\sum_{j,(i,j)\\in E}x_{ij} \\leq 1,

    where :math:`E` are the edges of the graph and :math:`x_{ij}` is a binary number that selects
    whether to include the edge :math:`(i, j)`.

    A set of edges satisfies the out-flow constraint whenever the following Hamiltonian is minimized:

    .. math::

        \\sum_{i\\in V}\\left(d_{i}^{out}(d_{i}^{out} - 2)\\mathbb{I}
        - 2(d_{i}^{out}-1)\\sum_{j,(i,j)\\in E}\\hat{Z}_{ij} +
        \\left( \\sum_{j,(i,j)\\in E}\\hat{Z}_{ij} \\right)^{2}\\right)


    where :math:`V` are the graph vertices, :math:`d_{i}^{\\rm out}` is the outdegree of node
    :math:`i`, and :math:`Z_{ij}` is a qubit Pauli-Z matrix acting
    upon the qubit specified by the pair :math:`(i, j)`. Mapping from edges to wires can be achieved
    using :func:`~.edges_to_wires`.

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges

    Returns:
        qml.Hamiltonian: the out flow constraint Hamiltonian

    Raises:
        ValueError: if the input graph is not directed
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(f'Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}')
    if isinstance(graph, (nx.DiGraph, rx.PyDiGraph)) and (not hasattr(graph, 'out_edges')):
        raise ValueError('Input graph must be directed')
    hamiltonian = Hamiltonian([], [])
    graph_nodes = graph.node_indexes() if isinstance(graph, rx.PyDiGraph) else graph.nodes
    for node in graph_nodes:
        hamiltonian += _inner_out_flow_constraint_hamiltonian(graph, node)
    return hamiltonian