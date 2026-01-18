import networkx as nx
from networkx.utils import np_random_state
def circular_layout(G, scale=1, center=None, dim=2):
    """Position nodes on a circle.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.
        If dim>2, the remaining dimensions are set to zero
        in the returned positions.
        If dim<2, a ValueError is raised.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    ValueError
        If dim < 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.circular_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np
    if dim < 2:
        raise ValueError('cannot handle dimensions < 2')
    G, center = _process_params(G, center, dim)
    paddims = max(0, dim - 2)
    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {nx.utils.arbitrary_element(G): center}
    else:
        theta = np.linspace(0, 1, len(G) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack([np.cos(theta), np.sin(theta), np.zeros((len(G), paddims))])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))
    return pos