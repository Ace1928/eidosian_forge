import warnings
import networkx as nx
@nx._dispatch
def single_target_shortest_path(G, target, cutoff=None):
    """Compute shortest path to target from all nodes that reach target.

    Parameters
    ----------
    G : NetworkX graph

    target : node label
       Target node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    paths : dictionary
        Dictionary, keyed by target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> path = nx.single_target_shortest_path(G, 4)
    >>> path[0]
    [0, 1, 2, 3, 4]

    Notes
    -----
    The shortest path is not necessarily unique. So there can be multiple
    paths between the source and each target node, all of which have the
    same 'shortest' length. For each target node, this function returns
    only one of those paths.

    See Also
    --------
    shortest_path, single_source_shortest_path
    """
    if target not in G:
        raise nx.NodeNotFound(f'Target {target} not in G')

    def join(p1, p2):
        return p2 + p1
    adj = G.pred if G.is_directed() else G.adj
    if cutoff is None:
        cutoff = float('inf')
    nextlevel = {target: 1}
    paths = {target: [target]}
    return dict(_single_shortest_path(adj, nextlevel, paths, cutoff, join))