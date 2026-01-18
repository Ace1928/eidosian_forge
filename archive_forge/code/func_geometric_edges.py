import math
from bisect import bisect_left
from itertools import accumulate, combinations, product
import networkx as nx
from networkx.utils import py_random_state
@nx._dispatch(node_attrs='pos_name')
def geometric_edges(G, radius, p=2, *, pos_name='pos'):
    """Returns edge list of node pairs within `radius` of each other.

    Parameters
    ----------
    G : networkx graph
        The graph from which to generate the edge list. The nodes in `G` should
        have an attribute ``pos`` corresponding to the node position, which is
        used to compute the distance to other nodes.
    radius : scalar
        The distance threshold. Edges are included in the edge list if the
        distance between the two nodes is less than `radius`.
    pos_name : string, default="pos"
        The name of the node attribute which represents the position of each
        node in 2D coordinates. Every node in the Graph must have this attribute.
    p : scalar, default=2
        The `Minkowski distance metric
        <https://en.wikipedia.org/wiki/Minkowski_distance>`_ used to compute
        distances. The default value is 2, i.e. Euclidean distance.

    Returns
    -------
    edges : list
        List of edges whose distances are less than `radius`

    Notes
    -----
    Radius uses Minkowski distance metric `p`.
    If scipy is available, `scipy.spatial.cKDTree` is used to speed computation.

    Examples
    --------
    Create a graph with nodes that have a "pos" attribute representing 2D
    coordinates.

    >>> G = nx.Graph()
    >>> G.add_nodes_from([
    ...     (0, {"pos": (0, 0)}),
    ...     (1, {"pos": (3, 0)}),
    ...     (2, {"pos": (8, 0)}),
    ... ])
    >>> nx.geometric_edges(G, radius=1)
    []
    >>> nx.geometric_edges(G, radius=4)
    [(0, 1)]
    >>> nx.geometric_edges(G, radius=6)
    [(0, 1), (1, 2)]
    >>> nx.geometric_edges(G, radius=9)
    [(0, 1), (0, 2), (1, 2)]
    """
    for n, pos in G.nodes(data=pos_name):
        if pos is None:
            raise nx.NetworkXError(f"Node {n} (and all nodes) must have a '{pos_name}' attribute.")
    return _geometric_edges(G, radius, p, pos_name)