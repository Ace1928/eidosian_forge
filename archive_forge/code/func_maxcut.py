from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane import qaoa
def maxcut(graph: Union[nx.Graph, rx.PyGraph]):
    """Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the
    MaxCut problem, for a given graph.

    The goal of the MaxCut problem for a particular graph is to find a partition of nodes into two sets,
    such that the number of edges in the graph with endpoints in different sets is maximized. Formally,
    we wish to find the `cut of the graph <https://en.wikipedia.org/wiki/Cut_(graph_theory)>`__ such
    that the number of edges crossing the cut is maximized.

    The MaxCut cost Hamiltonian is defined as:

    .. math:: H_C \\ = \\ \\frac{1}{2} \\displaystyle\\sum_{(i, j) \\in E(G)} \\big( Z_i Z_j \\ - \\ \\mathbb{I} \\big),

    where :math:`G` is a graph, :math:`\\mathbb{I}` is the identity, and :math:`Z_i` and :math:`Z_j` are
    the Pauli-Z operators on the :math:`i`-th and :math:`j`-th wire respectively.

    The mixer Hamiltonian returned from :func:`~qaoa.maxcut` is :func:`~qaoa.x_mixer` applied to all wires.

    .. note::

        **Recommended initialization circuit:**
            Even superposition over all basis states

    Args:
        graph (nx.Graph or rx.PyGraph): a graph defining the pairs of wires on which each term of the Hamiltonian acts

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    **Example**

    >>> import networkx as nx
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
      (-1.0) [I0]
    + (0.5) [Z0 Z1]
    + (0.5) [Z1 Z2]
    >>> print(mixer_h)
      (1) [X0]
    + (1) [X1]
    + (1) [X2]

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1,""), (1,2,"")])
    >>> cost_h, mixer_h = qml.qaoa.maxcut(graph)
    >>> print(cost_h)
      (-1.0) [I0]
    + (0.5) [Z0 Z1]
    + (0.5) [Z1 Z2]
    >>> print(mixer_h)
      (1) [X0]
    + (1) [X1]
    + (1) [X2]
    """
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(f'Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}')
    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.nodes()
    graph_edges = sorted(graph.edge_list()) if is_rx else graph.edges
    get_nvalue = lambda i: graph_nodes[i] if is_rx else i
    identity_h = qml.Hamiltonian([-0.5 for e in graph_edges], [qml.Identity(get_nvalue(e[0])) @ qml.Identity(get_nvalue(e[1])) for e in graph_edges])
    H = edge_driver(graph, ['10', '01']) + identity_h
    H.grouping_indices = [list(range(len(H.ops)))]
    return (H, qaoa.x_mixer(graph_nodes))