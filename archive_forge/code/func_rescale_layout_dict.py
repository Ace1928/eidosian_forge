import networkx as nx
from networkx.utils import np_random_state
def rescale_layout_dict(pos, scale=1):
    """Return a dictionary of scaled positions keyed by node

    Parameters
    ----------
    pos : A dictionary of positions keyed by node

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : A dictionary of positions keyed by node

    Examples
    --------
    >>> import numpy as np
    >>> pos = {0: np.array((0, 0)), 1: np.array((1, 1)), 2: np.array((0.5, 0.5))}
    >>> nx.rescale_layout_dict(pos)
    {0: array([-1., -1.]), 1: array([1., 1.]), 2: array([0., 0.])}

    >>> pos = {0: np.array((0, 0)), 1: np.array((-1, 1)), 2: np.array((-0.5, 0.5))}
    >>> nx.rescale_layout_dict(pos, scale=2)
    {0: array([ 2., -2.]), 1: array([-2.,  2.]), 2: array([0., 0.])}

    See Also
    --------
    rescale_layout
    """
    import numpy as np
    if not pos:
        return {}
    pos_v = np.array(list(pos.values()))
    pos_v = rescale_layout(pos_v, scale=scale)
    return dict(zip(pos, pos_v))