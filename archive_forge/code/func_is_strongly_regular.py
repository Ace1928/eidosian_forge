import networkx as nx
from networkx.utils import not_implemented_for
from .distance_measures import diameter
@not_implemented_for('directed', 'multigraph')
@nx._dispatch
def is_strongly_regular(G):
    """Returns True if and only if the given graph is strongly
    regular.

    An undirected graph is *strongly regular* if

    * it is regular,
    * each pair of adjacent vertices has the same number of neighbors in
      common,
    * each pair of nonadjacent vertices has the same number of neighbors
      in common.

    Each strongly regular graph is a distance-regular graph.
    Conversely, if a distance-regular graph has diameter two, then it is
    a strongly regular graph. For more information on distance-regular
    graphs, see :func:`is_distance_regular`.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    bool
        Whether `G` is strongly regular.

    Examples
    --------

    The cycle graph on five vertices is strongly regular. It is
    two-regular, each pair of adjacent vertices has no shared neighbors,
    and each pair of nonadjacent vertices has one shared neighbor::

        >>> G = nx.cycle_graph(5)
        >>> nx.is_strongly_regular(G)
        True

    """
    return is_distance_regular(G) and diameter(G) == 2