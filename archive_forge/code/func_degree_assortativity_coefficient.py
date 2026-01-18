import networkx as nx
from networkx.algorithms.assortativity.mixing import (
from networkx.algorithms.assortativity.pairs import node_degree_xy
@nx._dispatch(edge_attrs='weight')
def degree_assortativity_coefficient(G, x='out', y='in', weight=None, nodes=None):
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute degree assortativity only for nodes in container.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> r = nx.degree_assortativity_coefficient(G)
    >>> print(f"{r:3.1f}")
    -0.5

    See Also
    --------
    attribute_assortativity_coefficient
    numeric_assortativity_coefficient
    degree_mixing_dict
    degree_mixing_matrix

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , where e is the joint
    probability distribution (mixing matrix) of the degrees.  If G is
    directed than the matrix e is the joint probability of the
    user-specified degree type for the source and target.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
    if nodes is None:
        nodes = G.nodes
    degrees = None
    if G.is_directed():
        indeg = {d for _, d in G.in_degree(nodes, weight=weight)} if 'in' in (x, y) else set()
        outdeg = {d for _, d in G.out_degree(nodes, weight=weight)} if 'out' in (x, y) else set()
        degrees = set.union(indeg, outdeg)
    else:
        degrees = {d for _, d in G.degree(nodes, weight=weight)}
    mapping = {d: i for i, d in enumerate(degrees)}
    M = degree_mixing_matrix(G, x=x, y=y, nodes=nodes, weight=weight, mapping=mapping)
    return _numeric_ac(M, mapping=mapping)