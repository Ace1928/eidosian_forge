from functools import partial
import networkx as nx
from networkx.utils import (
@not_implemented_for('directed')
@np_random_state(5)
@nx._dispatch(edge_attrs='weight')
def algebraic_connectivity(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    """Returns the algebraic connectivity of an undirected graph.

    The algebraic connectivity of a connected undirected graph is the second
    smallest eigenvalue of its Laplacian matrix.

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
    algebraic_connectivity : float
        Algebraic connectivity.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    NetworkXError
        If G has less than two nodes.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    For undirected graphs algebraic connectivity can tell us if a graph is connected or not
    `G` is connected iff  ``algebraic_connectivity(G) > 0``:

    >>> G = nx.complete_graph(5)
    >>> nx.algebraic_connectivity(G) > 0
    True
    >>> G.add_node(10)  # G is no longer connected
    >>> nx.algebraic_connectivity(G) > 0
    False

    """
    if len(G) < 2:
        raise nx.NetworkXError('graph has less than two nodes.')
    G = _preprocess_graph(G, weight)
    if not nx.is_connected(G):
        return 0.0
    L = nx.laplacian_matrix(G)
    if L.shape[0] == 2:
        return 2.0 * L[0, 0] if not normalized else 2.0
    find_fiedler = _get_fiedler_func(method)
    x = None if method != 'lobpcg' else _rcm_estimate(G, G)
    sigma, fiedler = find_fiedler(L, x, normalized, tol, seed)
    return sigma