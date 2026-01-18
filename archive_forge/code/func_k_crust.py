import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
@nx._dispatch(preserve_all_attrs=True)
def k_crust(G, k=None, core_number=None):
    """Returns the k-crust of G.

    The k-crust is the graph G with the edges of the k-core removed
    and isolated nodes found after the removal of edges are also removed.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph.
    k : int, optional
      The order of the shell.  If not specified return the main crust.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
       The k-crust subgraph

    Raises
    ------
    NetworkXError
        The k-crust is not implemented for graphs with self loops
        or parallel edges.

    Notes
    -----
    This definition of k-crust is different than the definition in [1]_.
    The k-crust in [1]_ is equivalent to the k+1 crust of this algorithm.

    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    See Also
    --------
    core_number

    References
    ----------
    .. [1] A model of Internet topology using k-shell decomposition
       Shai Carmi, Shlomo Havlin, Scott Kirkpatrick, Yuval Shavitt,
       and Eran Shir, PNAS  July 3, 2007   vol. 104  no. 27  11150-11154
       http://www.pnas.org/content/104/27/11150.full
    """
    if core_number is None:
        core_number = nx.core_number(G)
    if k is None:
        k = max(core_number.values()) - 1
    nodes = (v for v in core_number if core_number[v] <= k)
    return G.subgraph(nodes).copy()