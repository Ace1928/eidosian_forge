import numbers
from collections import Counter
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import discrete_sequence, py_random_state, weighted_choice
@py_random_state(6)
@nx._dispatch(graphs=None)
def scale_free_graph(n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, seed=None, initial_graph=None):
    """Returns a scale-free directed graph.

    Parameters
    ----------
    n : integer
        Number of nodes in graph
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution.
    beta : float
        Probability for adding an edge between two existing nodes.
        One existing node is chosen randomly according the in-degree
        distribution and the other chosen randomly according to the out-degree
        distribution.
    gamma : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the out-degree distribution.
    delta_in : float
        Bias for choosing nodes from in-degree distribution.
    delta_out : float
        Bias for choosing nodes from out-degree distribution.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    initial_graph : MultiDiGraph instance, optional
        Build the scale-free graph starting from this initial MultiDiGraph,
        if provided.

    Returns
    -------
    MultiDiGraph

    Examples
    --------
    Create a scale-free graph on one hundred nodes::

    >>> G = nx.scale_free_graph(100)

    Notes
    -----
    The sum of `alpha`, `beta`, and `gamma` must be 1.

    References
    ----------
    .. [1] B. Bollobás, C. Borgs, J. Chayes, and O. Riordan,
           Directed scale-free graphs,
           Proceedings of the fourteenth annual ACM-SIAM Symposium on
           Discrete Algorithms, 132--139, 2003.
    """

    def _choose_node(candidates, node_list, delta):
        if delta > 0:
            bias_sum = len(node_list) * delta
            p_delta = bias_sum / (bias_sum + len(candidates))
            if seed.random() < p_delta:
                return seed.choice(node_list)
        return seed.choice(candidates)
    if initial_graph is not None and hasattr(initial_graph, '_adj'):
        if not isinstance(initial_graph, nx.MultiDiGraph):
            raise nx.NetworkXError('initial_graph must be a MultiDiGraph.')
        G = initial_graph
    else:
        G = nx.MultiDiGraph([(0, 1), (1, 2), (2, 0)])
    if alpha <= 0:
        raise ValueError('alpha must be > 0.')
    if beta <= 0:
        raise ValueError('beta must be > 0.')
    if gamma <= 0:
        raise ValueError('gamma must be > 0.')
    if abs(alpha + beta + gamma - 1.0) >= 1e-09:
        raise ValueError('alpha+beta+gamma must equal 1.')
    if delta_in < 0:
        raise ValueError('delta_in must be >= 0.')
    if delta_out < 0:
        raise ValueError('delta_out must be >= 0.')
    vs = sum((count * [idx] for idx, count in G.out_degree()), [])
    ws = sum((count * [idx] for idx, count in G.in_degree()), [])
    node_list = list(G.nodes())
    numeric_nodes = [n for n in node_list if isinstance(n, numbers.Number)]
    if len(numeric_nodes) > 0:
        cursor = max((int(n.real) for n in numeric_nodes)) + 1
    else:
        cursor = 0
    while len(G) < n:
        r = seed.random()
        if r < alpha:
            v = cursor
            cursor += 1
            node_list.append(v)
            w = _choose_node(ws, node_list, delta_in)
        elif r < alpha + beta:
            v = _choose_node(vs, node_list, delta_out)
            w = _choose_node(ws, node_list, delta_in)
        else:
            v = _choose_node(vs, node_list, delta_out)
            w = cursor
            cursor += 1
            node_list.append(w)
        G.add_edge(v, w)
        vs.append(v)
        ws.append(w)
    return G