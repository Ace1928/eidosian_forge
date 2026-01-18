import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
@nx._dispatch(preserve_all_attrs=True)
def k_core(G, k=None, core_number=None):
    """Returns the k-core of G.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph
    k : int, optional
      The order of the core.  If not specified return the main core.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
      The k-core subgraph

    Raises
    ------
    NetworkXError
      The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    The main core is the core with the largest degree.

    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    See Also
    --------
    core_number

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik,  2003.
       https://arxiv.org/abs/cs.DS/0310049
    """

    def k_filter(v, k, c):
        return c[v] >= k
    return _core_subgraph(G, k_filter, k, core_number)