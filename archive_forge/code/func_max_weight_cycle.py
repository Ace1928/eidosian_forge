from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane import qaoa
def max_weight_cycle(graph: Union[nx.Graph, rx.PyGraph, rx.PyDiGraph], constrained: bool=True):
    """Returns the QAOA cost Hamiltonian and the recommended mixer corresponding to the
    maximum-weighted cycle problem, for a given graph.

    The maximum-weighted cycle problem is defined in the following way (see
    `here <https://1qbit.com/whitepaper/arbitrage/>`__ for more details).
    The product of weights of a subset of edges in a graph is given by

    .. math:: P = \\prod_{(i, j) \\in E} [(c_{ij} - 1)x_{ij} + 1]

    where :math:`E` are the edges of the graph, :math:`x_{ij}` is a binary number that selects
    whether to include the edge :math:`(i, j)` and :math:`c_{ij}` is the corresponding edge weight.
    Our objective is to maximimize :math:`P`, subject to selecting the :math:`x_{ij}` so that
    our subset of edges composes a `cycle <https://en.wikipedia.org/wiki/Cycle_(graph_theory)>`__.

    Args:
        graph (nx.Graph or rx.PyGraph or rx.PyDiGraph): the directed graph on which the Hamiltonians are defined
        constrained (bool): specifies the variant of QAOA that is performed (constrained or unconstrained)

    Returns:
        (.Hamiltonian, .Hamiltonian, dict): The cost and mixer Hamiltonians, as well as a dictionary
        mapping from wires to the graph's edges

    .. details::
        :title: Usage Details

        There are two variations of QAOA for this problem, constrained and unconstrained:

        **Constrained**

        .. note::

            This method of constrained QAOA was introduced by Hadfield, Wang, Gorman, Rieffel,
            Venturelli, and Biswas in `arXiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__.

        The maximum weighted cycle cost Hamiltonian for unconstrained QAOA is

        .. math:: H_C = H_{\\rm loss}.

        Here, :math:`H_{\\rm loss}` is a loss Hamiltonian:

        .. math:: H_{\\rm loss} = \\sum_{(i, j) \\in E} Z_{ij}\\log c_{ij}

        where :math:`E` are the edges of the graph and :math:`Z_{ij}` is a qubit Pauli-Z matrix
        acting upon the wire specified by the edge :math:`(i, j)` (see :func:`~.loss_hamiltonian`
        for more details).

        The returned mixer Hamiltonian is :func:`~.cycle_mixer` given by

        .. math:: H_M = \\frac{1}{4}\\sum_{(i, j)\\in E}
                \\left(\\sum_{k \\in V, k\\neq i, k\\neq j, (i, k) \\in E, (k, j) \\in E}
                \\left[X_{ij}X_{ik}X_{kj} +Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\\right]
                \\right).

        This mixer provides transitions between collections of cycles, i.e., any subset of edges
        in :math:`E` such that all the graph's nodes :math:`V` have zero net flow
        (see the :func:`~.net_flow_constraint` function).

        .. note::

            **Recommended initialization circuit:**
                Your circuit must prepare a state that corresponds to a cycle (or a superposition
                of cycles). Follow the example code below to see how this is done.

        **Unconstrained**

        The maximum weighted cycle cost Hamiltonian for constrained QAOA is defined as:

        .. math:: H_C \\ = H_{\\rm loss} + 3 H_{\\rm netflow} + 3 H_{\\rm outflow}.

        The netflow constraint Hamiltonian :func:`~.net_flow_constraint` is given by

        .. math:: H_{\\rm netflow} = \\sum_{i \\in V} \\left((d_{i}^{\\rm out} - d_{i}^{\\rm in})\\mathbb{I} -
                \\sum_{j, (i, j) \\in E} Z_{ij} + \\sum_{j, (j, i) \\in E} Z_{ji} \\right)^{2},

        where :math:`d_{i}^{\\rm out}` and :math:`d_{i}^{\\rm in}` are
        the outdegree and indegree, respectively, of node :math:`i`. It is minimized whenever a
        subset of edges in :math:`E` results in zero net flow from each node in :math:`V`.

        The outflow constraint Hamiltonian :func:`~.out_flow_constraint` is given by

        .. math:: H_{\\rm outflow} = \\sum_{i\\in V}\\left(d_{i}^{out}(d_{i}^{out} - 2)\\mathbb{I}
                - 2(d_{i}^{out}-1)\\sum_{j,(i,j)\\in E}\\hat{Z}_{ij} +
                \\left( \\sum_{j,(i,j)\\in E}\\hat{Z}_{ij} \\right)^{2}\\right).

        It is minimized whenever a subset of edges in :math:`E` results in an outflow of at most one
        from each node in :math:`V`.

        The returned mixer Hamiltonian is :func:`~.x_mixer` applied to all wires.

        .. note::

            **Recommended initialization circuit:**
                Even superposition over all basis states.

        **Example**

        First set up a simple graph:

        .. code-block:: python

            import pennylane as qml
            import numpy as np
            import networkx as nx

            a = np.random.random((4, 4))
            np.fill_diagonal(a, 0)
            g = nx.DiGraph(a)

        The cost and mixer Hamiltonian as well as the mapping from wires to edges can be loaded
        using:

        >>> cost, mixer, mapping = qml.qaoa.max_weight_cycle(g, constrained=True)

        Since we are using ``constrained=True``, we must ensure that the input state to the QAOA
        algorithm corresponds to a cycle. Consider the mapping:

        >>> mapping
        {0: (0, 1),
         1: (0, 2),
         2: (0, 3),
         3: (1, 0),
         4: (1, 2),
         5: (1, 3),
         6: (2, 0),
         7: (2, 1),
         8: (2, 3),
         9: (3, 0),
         10: (3, 1),
         11: (3, 2)}

        A simple cycle is given by the edges ``(0, 1)`` and ``(1, 0)`` and corresponding wires
        ``0`` and ``3``. Hence, the state :math:`|100100000000\\rangle` corresponds to a cycle and
        can be prepared using :class:`~.BasisState` or simple :class:`~.PauliX` rotations on the
        ``0`` and ``3`` wires.
    """
    if not isinstance(graph, (nx.Graph, rx.PyGraph, rx.PyDiGraph)):
        raise ValueError(f'Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph, got {type(graph).__name__}')
    mapping = qaoa.cycle.wires_to_edges(graph)
    if constrained:
        cost_h = qaoa.cycle.loss_hamiltonian(graph)
        cost_h.grouping_indices = [list(range(len(cost_h.ops)))]
        return (cost_h, qaoa.cycle.cycle_mixer(graph), mapping)
    cost_h = qaoa.cycle.loss_hamiltonian(graph) + 3 * (qaoa.cycle.net_flow_constraint(graph) + qaoa.cycle.out_flow_constraint(graph))
    mixer_h = qaoa.x_mixer(mapping.keys())
    return (cost_h, mixer_h, mapping)