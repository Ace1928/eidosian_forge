from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def is_weighted(G, edge=None, weight='weight'):
    """Returns True if `G` has weighted edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    edge : tuple, optional
        A 2-tuple specifying the only edge in `G` that will be tested. If
        None, then every edge in `G` is tested.

    weight: string, optional
        The attribute name used to query for edge weights.

    Returns
    -------
    bool
        A boolean signifying if `G`, or the specified edge, is weighted.

    Raises
    ------
    NetworkXError
        If the specified edge does not exist.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.is_weighted(G)
    False
    >>> nx.is_weighted(G, (2, 3))
    False

    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2, weight=1)
    >>> nx.is_weighted(G)
    True

    """
    if edge is not None:
        data = G.get_edge_data(*edge)
        if data is None:
            msg = f'Edge {edge!r} does not exist.'
            raise nx.NetworkXError(msg)
        return weight in data
    if is_empty(G):
        return False
    return all((weight in data for u, v, data in G.edges(data=True)))