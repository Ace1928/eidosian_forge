from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane import qaoa
def max_independent_set(graph: Union[nx.Graph, rx.PyGraph], constrained: bool=True):
    """For a given graph, returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the Maximum Independent Set problem.

    Given some graph :math:`G`, an independent set is a set of vertices such that no pair of vertices in the set
    share a common edge. The Maximum Independent Set problem, is the problem of finding the largest such set.

    Args:
        graph (nx.Graph or rx.PyGraph): a graph whose edges define the pairs of vertices on which each term of the Hamiltonian acts
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian): The cost and mixer Hamiltonians

    .. details::
        :title: Usage Details

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by
            `Hadfield, Wang, Gorman, Rieffel, Venturelli, and Biswas (2019) <https://doi.org/10.3390/a12020034>`__.

        The Maximum Independent Set cost Hamiltonian for constrained QAOA is defined as:

        .. math:: H_C \\ = \\ \\displaystyle\\sum_{v \\in V(G)} Z_{v},

        where :math:`V(G)` is the set of vertices of the input graph, and :math:`Z_i` is the Pauli-Z
        operator applied to the :math:`i`-th vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.bit_flip_mixer` applied to :math:`G`.

        .. note::

            **Recommended initialization circuit:**
                Each wire in the :math:`|0\\rangle` state.

        **Unconstrained**

        The Maximum Independent Set cost Hamiltonian for unconstrained QAOA is defined as:

        .. math:: H_C \\ = \\ 3 \\sum_{(i, j) \\in E(G)} (Z_i Z_j \\ - \\ Z_i \\ - \\ Z_j) \\ + \\
                  \\displaystyle\\sum_{i \\in V(G)} Z_i

        where :math:`E(G)` is the set of edges of :math:`G`, :math:`V(G)` is the set of vertices,
        and :math:`Z_i` is the Pauli-Z operator acting on the :math:`i`-th vertex.

        The returned mixer Hamiltonian is :func:`~qaoa.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states.

    """
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(f'Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}')
    graph_nodes = graph.nodes()
    if constrained:
        cost_h = bit_driver(graph_nodes, 1)
        cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
        return (cost_h, qaoa.bit_flip_mixer(graph, 0))
    cost_h = 3 * edge_driver(graph, ['10', '01', '00']) + bit_driver(graph_nodes, 1)
    mixer_h = qaoa.x_mixer(graph_nodes)
    cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
    return (cost_h, mixer_h)