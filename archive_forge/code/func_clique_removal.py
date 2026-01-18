import networkx as nx
from networkx.algorithms.approximation import ramsey
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def clique_removal(G):
    """Repeatedly remove cliques from the graph.

    Results in a $O(|V|/(\\log |V|)^2)$ approximation of maximum clique
    and independent set. Returns the largest independent set found, along
    with found maximal cliques.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    max_ind_cliques : (set, list) tuple
        2-tuple of Maximal Independent Set and list of maximal cliques (sets).

    Examples
    --------
    >>> G = nx.path_graph(10)
    >>> nx.approximation.clique_removal(G)
    ({0, 2, 4, 6, 9}, [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}])

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Boppana, R., & Halldórsson, M. M. (1992).
        Approximating maximum independent sets by excluding subgraphs.
        BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    graph = G.copy()
    c_i, i_i = ramsey.ramsey_R2(graph)
    cliques = [c_i]
    isets = [i_i]
    while graph:
        graph.remove_nodes_from(c_i)
        c_i, i_i = ramsey.ramsey_R2(graph)
        if c_i:
            cliques.append(c_i)
        if i_i:
            isets.append(i_i)
    maxiset = max(isets, key=len)
    return (maxiset, cliques)