from functools import partial
import networkx as nx
from networkx.utils import (
@not_implemented_for('directed')
@np_random_state(5)
@nx._dispatch(edge_attrs='weight')
def fiedler_vector(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    """Returns the Fiedler vector of a connected undirected graph.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    fiedler_vector : NumPy array of floats.
        Fiedler vector.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    NetworkXError
        If G has less than two nodes or is not connected.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    Given a connected graph the signs of the values in the Fiedler vector can be
    used to partition the graph into two components.

    >>> G = nx.barbell_graph(5, 0)
    >>> nx.fiedler_vector(G, normalized=True, seed=1)
    array([-0.32864129, -0.32864129, -0.32864129, -0.32864129, -0.26072899,
            0.26072899,  0.32864129,  0.32864129,  0.32864129,  0.32864129])

    The connected components are the two 5-node cliques of the barbell graph.
    """
    import numpy as np
    if len(G) < 2:
        raise nx.NetworkXError('graph has less than two nodes.')
    G = _preprocess_graph(G, weight)
    if not nx.is_connected(G):
        raise nx.NetworkXError('graph is not connected.')
    if len(G) == 2:
        return np.array([1.0, -1.0])
    find_fiedler = _get_fiedler_func(method)
    L = nx.laplacian_matrix(G)
    x = None if method != 'lobpcg' else _rcm_estimate(G, G)
    sigma, fiedler = find_fiedler(L, x, normalized, tol, seed)
    return fiedler