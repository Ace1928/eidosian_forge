import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(edge_attrs='weight')
def periphery(G, e=None, usebounds=False, weight=None):
    """Returns the periphery of the graph G.

    The periphery is the set of nodes with eccentricity equal to the diameter.

    Parameters
    ----------
    G : NetworkX graph
       A graph

    e : eccentricity dictionary, optional
      A precomputed dictionary of eccentricities.

    weight : string, function, or None
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    p : list
       List of nodes in periphery

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (1, 4), (3, 4), (3, 5), (4, 5)])
    >>> nx.periphery(G)
    [2, 5]

    See Also
    --------
    barycenter
    center
    """
    if usebounds is True and e is None and (not G.is_directed()):
        return _extrema_bounding(G, compute='periphery', weight=weight)
    if e is None:
        e = eccentricity(G, weight=weight)
    diameter = max(e.values())
    p = [v for v in e if e[v] == diameter]
    return p