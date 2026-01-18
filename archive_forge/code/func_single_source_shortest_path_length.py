import warnings
import networkx as nx
@nx._dispatch
def single_source_shortest_path_length(G, source, cutoff=None):
    """Compute the shortest path lengths from source to all reachable nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : dict
        Dict keyed by node to shortest path length to source.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.single_source_shortest_path_length(G, 0)
    >>> length[4]
    4
    >>> for node in length:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4

    See Also
    --------
    shortest_path_length
    """
    if source not in G:
        raise nx.NodeNotFound(f'Source {source} is not in G')
    if cutoff is None:
        cutoff = float('inf')
    nextlevel = [source]
    return dict(_single_shortest_path_length(G._adj, nextlevel, cutoff))