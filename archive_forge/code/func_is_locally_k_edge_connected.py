import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def is_locally_k_edge_connected(G, s, t, k):
    """Tests to see if an edge in a graph is locally k-edge-connected.

    Is it impossible to disconnect s and t by removing fewer than k edges?
    If so, then s and t are locally k-edge-connected in G.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    s : node
        Source node

    t : node
        Target node

    k : integer
        local edge connectivity for nodes s and t

    Returns
    -------
    boolean
        True if s and t are locally k-edge-connected in G.

    See Also
    --------
    :func:`is_k_edge_connected`

    Examples
    --------
    >>> from networkx.algorithms.connectivity import is_locally_k_edge_connected
    >>> G = nx.barbell_graph(10, 0)
    >>> is_locally_k_edge_connected(G, 5, 15, k=1)
    True
    >>> is_locally_k_edge_connected(G, 5, 15, k=2)
    False
    >>> is_locally_k_edge_connected(G, 1, 5, k=2)
    True
    """
    if k < 1:
        raise ValueError(f'k must be positive, not {k}')
    if G.degree(s) < k or G.degree(t) < k:
        return False
    elif k == 1:
        return nx.has_path(G, s, t)
    else:
        localk = nx.connectivity.local_edge_connectivity(G, s, t, cutoff=k)
        return localk >= k