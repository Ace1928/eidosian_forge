import networkx as nx
from networkx.utils import np_random_state
def bipartite_layout(G, nodes, align='vertical', scale=1, center=None, aspect_ratio=4 / 3):
    """Position nodes in two straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nodes : list or container
        Nodes in one node set of the bipartite graph.
        This set will be placed on left or top.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    aspect_ratio : number (default=4/3):
        The ratio of the width to the height of the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)
    >>> top = nx.bipartite.sets(G)[0]
    >>> pos = nx.bipartite_layout(G, top)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np
    if align not in ('vertical', 'horizontal'):
        msg = 'align must be either vertical or horizontal.'
        raise ValueError(msg)
    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}
    height = 1
    width = aspect_ratio * height
    offset = (width / 2, height / 2)
    top = dict.fromkeys(nodes)
    bottom = [v for v in G if v not in top]
    nodes = list(top) + bottom
    left_xs = np.repeat(0, len(top))
    right_xs = np.repeat(width, len(bottom))
    left_ys = np.linspace(0, height, len(top))
    right_ys = np.linspace(0, height, len(bottom))
    top_pos = np.column_stack([left_xs, left_ys]) - offset
    bottom_pos = np.column_stack([right_xs, right_ys]) - offset
    pos = np.concatenate([top_pos, bottom_pos])
    pos = rescale_layout(pos, scale=scale) + center
    if align == 'horizontal':
        pos = pos[:, ::-1]
    pos = dict(zip(nodes, pos))
    return pos