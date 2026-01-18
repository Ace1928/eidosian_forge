from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def add_star(G_to_add_to, nodes_for_star, **attr):
    """Add a star to Graph G_to_add_to.

    The first node in `nodes_for_star` is the middle of the star.
    It is connected to all other nodes.

    Parameters
    ----------
    G_to_add_to : graph
        A NetworkX graph
    nodes_for_star : iterable container
        A container of nodes.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to every edge in star.

    See Also
    --------
    add_path, add_cycle

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_star(G, [0, 1, 2, 3])
    >>> nx.add_star(G, [10, 11, 12], weight=2)
    """
    nlist = iter(nodes_for_star)
    try:
        v = next(nlist)
    except StopIteration:
        return
    G_to_add_to.add_node(v)
    edges = ((v, n) for n in nlist)
    G_to_add_to.add_edges_from(edges, **attr)