import warnings
import networkx as nx
@nx._dispatch
def bidirectional_shortest_path(G, source, target):
    """Returns a list of nodes in a shortest path between source and target.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       starting node for path

    target : node label
       ending node for path

    Returns
    -------
    path: list
       List of nodes in a path from source to target.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3, 0, 4, 5, 6, 7, 4])
    >>> nx.bidirectional_shortest_path(G, 2, 6)
    [2, 1, 0, 4, 5, 6]

    See Also
    --------
    shortest_path

    Notes
    -----
    This algorithm is used by shortest_path(G, source, target).
    """
    if source not in G or target not in G:
        msg = f'Either source {source} or target {target} is not in G'
        raise nx.NodeNotFound(msg)
    results = _bidirectional_pred_succ(G, source, target)
    pred, succ, w = results
    path = []
    while w is not None:
        path.append(w)
        w = pred[w]
    path.reverse()
    w = succ[path[-1]]
    while w is not None:
        path.append(w)
        w = succ[w]
    return path