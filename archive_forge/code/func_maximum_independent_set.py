import networkx as nx
from networkx.algorithms.approximation import ramsey
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def maximum_independent_set(G):
    """Returns an approximate maximum independent set.

    Independent set or stable set is a set of vertices in a graph, no two of
    which are adjacent. That is, it is a set I of vertices such that for every
    two vertices in I, there is no edge connecting the two. Equivalently, each
    edge in the graph has at most one endpoint in I. The size of an independent
    set is the number of vertices it contains [1]_.

    A maximum independent set is a largest independent set for a given graph G
    and its size is denoted $\\alpha(G)$. The problem of finding such a set is called
    the maximum independent set problem and is an NP-hard optimization problem.
    As such, it is unlikely that there exists an efficient algorithm for finding
    a maximum independent set of a graph.

    The Independent Set algorithm is based on [2]_.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    iset : Set
        The apx-maximum independent set

    Examples
    --------
    >>> G = nx.path_graph(10)
    >>> nx.approximation.maximum_independent_set(G)
    {0, 2, 4, 6, 9}

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    Notes
    -----
    Finds the $O(|V|/(log|V|)^2)$ apx of independent set in the worst case.

    References
    ----------
    .. [1] `Wikipedia: Independent set
        <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_
    .. [2] Boppana, R., & Halldórsson, M. M. (1992).
       Approximating maximum independent sets by excluding subgraphs.
       BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    iset, _ = clique_removal(G)
    return iset