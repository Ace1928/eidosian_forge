import networkx as nx
from networkx.utils import np_random_state
def kamada_kawai_layout(G, dist=None, pos=None, weight='weight', scale=1, center=None, dim=2):
    """Position nodes using Kamada-Kawai path-length cost-function.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    dist : dict (default=None)
        A two-level dictionary of optimal distances between nodes,
        indexed by source and destination node.
        If None, the distance is computed using shortest_path_length().

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        circular_layout() for dim >= 2 and a linear layout for dim == 1.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.kamada_kawai_layout(G)
    """
    import numpy as np
    G, center = _process_params(G, center, dim)
    nNodes = len(G)
    if nNodes == 0:
        return {}
    if dist is None:
        dist = dict(nx.shortest_path_length(G, weight=weight))
    dist_mtx = 1000000.0 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]
    if pos is None:
        if dim >= 3:
            pos = random_layout(G, dim=dim)
        elif dim == 2:
            pos = circular_layout(G, dim=dim)
        else:
            pos = dict(zip(G, np.linspace(0, 1, len(G))))
    pos_arr = np.array([pos[n] for n in G])
    pos = _kamada_kawai_solve(dist_mtx, pos_arr, dim)
    pos = rescale_layout(pos, scale=scale) + center
    return dict(zip(G, pos))