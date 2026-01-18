import networkx as nx
from networkx.utils import np_random_state
def bfs_layout(G, start, *, align='vertical', scale=1, center=None):
    """Position nodes according to breadth-first search algorithm.

    Parameters
    ----------
    G : NetworkX graph
        A position will be assigned to every node in G.

    start : node in `G`
        Starting node for bfs

    center : array-like or None
        Coordinate pair around which to center the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.bfs_layout(G, 0)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    G, center = _process_params(G, center, 2)
    layers = dict(enumerate(nx.bfs_layers(G, start)))
    if len(G) != sum((len(nodes) for nodes in layers.values())):
        raise nx.NetworkXError("bfs_layout didn't include all nodes. Perhaps use input graph:\n        G.subgraph(nx.node_connected_component(G, start))")
    return multipartite_layout(G, subset_key=layers, align=align, scale=scale, center=center)