import networkx as nx
from networkx.utils import np_random_state
def spectral_layout(G, weight='weight', scale=1, center=None, dim=2):
    """Position nodes using the eigenvectors of the graph Laplacian.

    Using the unnormalized Laplacian, the layout shows possible clusters of
    nodes which are an approximation of the ratio cut. If dim is the number of
    dimensions then the positions are the entries of the dim eigenvectors
    corresponding to the ascending eigenvalues starting from the second one.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

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
    >>> pos = nx.spectral_layout(G)

    Notes
    -----
    Directed graphs will be considered as undirected graphs when
    positioning the nodes.

    For larger graphs (>500 nodes) this will use the SciPy sparse
    eigenvalue solver (ARPACK).
    """
    import numpy as np
    G, center = _process_params(G, center, dim)
    if len(G) <= 2:
        if len(G) == 0:
            pos = np.array([])
        elif len(G) == 1:
            pos = np.array([center])
        else:
            pos = np.array([np.zeros(dim), np.array(center) * 2.0])
        return dict(zip(G, pos))
    try:
        if len(G) < 500:
            raise ValueError
        A = nx.to_scipy_sparse_array(G, weight=weight, dtype='d')
        if G.is_directed():
            A = A + np.transpose(A)
        pos = _sparse_spectral(A, dim)
    except (ImportError, ValueError):
        A = nx.to_numpy_array(G, weight=weight)
        if G.is_directed():
            A += A.T
        pos = _spectral(A, dim)
    pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos