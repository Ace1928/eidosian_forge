import networkx as nx
from networkx.utils import py_random_state
@py_random_state(1)
@nx._dispatch(graphs=None)
def joint_degree_graph(joint_degrees, seed=None):
    """Generates a random simple graph with the given joint degree dictionary.

    Parameters
    ----------
    joint_degrees :  dictionary of dictionary of integers
        A joint degree dictionary in which entry ``joint_degrees[k][l]`` is the
        number of edges joining nodes of degree *k* with nodes of degree *l*.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph
        A graph with the specified joint degree dictionary.

    Raises
    ------
    NetworkXError
        If *joint_degrees* dictionary is not realizable.

    Notes
    -----
    In each iteration of the "while loop" the algorithm picks two disconnected
    nodes *v* and *w*, of degree *k* and *l* correspondingly,  for which
    ``joint_degrees[k][l]`` has not reached its target yet. It then adds
    edge (*v*, *w*) and increases the number of edges in graph G by one.

    The intelligence of the algorithm lies in the fact that  it is always
    possible to add an edge between such disconnected nodes *v* and *w*,
    even if one or both nodes do not have free stubs. That is made possible by
    executing a "neighbor switch", an edge rewiring move that releases
    a free stub while keeping the joint degree of G the same.

    The algorithm continues for E (number of edges) iterations of
    the "while loop", at the which point all entries of the given
    ``joint_degrees[k][l]`` have reached their target values and the
    construction is complete.

    References
    ----------
    ..  [1] M. Gjoka, B. Tillman, A. Markopoulou, "Construction of Simple
        Graphs with a Target Joint Degree Matrix and Beyond", IEEE Infocom, '15

    Examples
    --------
    >>> joint_degrees = {
    ...     1: {4: 1},
    ...     2: {2: 2, 3: 2, 4: 2},
    ...     3: {2: 2, 4: 1},
    ...     4: {1: 1, 2: 2, 3: 1},
    ... }
    >>> G = nx.joint_degree_graph(joint_degrees)
    >>>
    """
    if not is_valid_joint_degree(joint_degrees):
        msg = 'Input joint degree dict not realizable as a simple graph'
        raise nx.NetworkXError(msg)
    degree_count = {k: sum(l.values()) // k for k, l in joint_degrees.items() if k > 0}
    N = sum(degree_count.values())
    G = nx.empty_graph(N)
    h_degree_nodelist = {}
    h_node_residual = {}
    nodeid = 0
    for degree, num_nodes in degree_count.items():
        h_degree_nodelist[degree] = range(nodeid, nodeid + num_nodes)
        for v in h_degree_nodelist[degree]:
            h_node_residual[v] = degree
        nodeid += int(num_nodes)
    for k in joint_degrees:
        for l in joint_degrees[k]:
            n_edges_add = joint_degrees[k][l]
            if n_edges_add > 0 and k >= l:
                k_size = degree_count[k]
                l_size = degree_count[l]
                k_nodes = h_degree_nodelist[k]
                l_nodes = h_degree_nodelist[l]
                k_unsat = {v for v in k_nodes if h_node_residual[v] > 0}
                if k != l:
                    l_unsat = {w for w in l_nodes if h_node_residual[w] > 0}
                else:
                    l_unsat = k_unsat
                    n_edges_add = joint_degrees[k][l] // 2
                while n_edges_add > 0:
                    v = k_nodes[seed.randrange(k_size)]
                    w = l_nodes[seed.randrange(l_size)]
                    if not G.has_edge(v, w) and v != w:
                        if h_node_residual[v] == 0:
                            _neighbor_switch(G, v, k_unsat, h_node_residual)
                        if h_node_residual[w] == 0:
                            if k != l:
                                _neighbor_switch(G, w, l_unsat, h_node_residual)
                            else:
                                _neighbor_switch(G, w, l_unsat, h_node_residual, avoid_node_id=v)
                        G.add_edge(v, w)
                        h_node_residual[v] -= 1
                        h_node_residual[w] -= 1
                        n_edges_add -= 1
                        if h_node_residual[v] == 0:
                            k_unsat.discard(v)
                        if h_node_residual[w] == 0:
                            l_unsat.discard(w)
    return G