import networkx as nx
from networkx.utils import np_random_state
def spiral_layout(G, scale=1, center=None, dim=2, resolution=0.35, equidistant=False):
    """Position nodes in a spiral layout.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.
    scale : number (default: 1)
        Scale factor for positions.
    center : array-like or None
        Coordinate pair around which to center the layout.
    dim : int, default=2
        Dimension of layout, currently only dim=2 is supported.
        Other dimension values result in a ValueError.
    resolution : float, default=0.35
        The compactness of the spiral layout returned.
        Lower values result in more compressed spiral layouts.
    equidistant : bool, default=False
        If True, nodes will be positioned equidistant from each other
        by decreasing angle further from center.
        If False, nodes will be positioned at equal angles
        from each other by increasing separation further from center.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    ValueError
        If dim != 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spiral_layout(G)
    >>> nx.draw(G, pos=pos)

    Notes
    -----
    This algorithm currently only works in two dimensions.

    """
    import numpy as np
    if dim != 2:
        raise ValueError('can only handle 2 dimensions')
    G, center = _process_params(G, center, dim)
    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G): center}
    pos = []
    if equidistant:
        chord = 1
        step = 0.5
        theta = resolution
        theta += chord / (step * theta)
        for _ in range(len(G)):
            r = step * theta
            theta += chord / r
            pos.append([np.cos(theta) * r, np.sin(theta) * r])
    else:
        dist = np.arange(len(G), dtype=float)
        angle = resolution * dist
        pos = np.transpose(dist * np.array([np.cos(angle), np.sin(angle)]))
    pos = rescale_layout(np.array(pos), scale=scale) + center
    pos = dict(zip(G, pos))
    return pos