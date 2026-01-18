from functools import partial
import networkx as nx
from networkx.utils import (
@nx._dispatch(edge_attrs='weight')
def spectral_bisection(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    """Bisect the graph using the Fiedler vector.

    This method uses the Fiedler vector to bisect a graph.
    The partition is defined by the nodes which are associated with
    either positive or negative values in the vector.

    Parameters
    ----------
    G : NetworkX Graph

    weight : str, optional (default: weight)
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
    bisection : tuple of sets
        Sets with the bisection of nodes

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.spectral_bisection(G)
    ({0, 1, 2}, {3, 4, 5})

    References
    ----------
    .. [1] M. E. J Newman 'Networks: An Introduction', pages 364-370
       Oxford University Press 2011.
    """
    import numpy as np
    v = nx.fiedler_vector(G, weight, normalized, tol, method, seed)
    nodes = np.array(list(G))
    pos_vals = v >= 0
    return (set(nodes[~pos_vals]), set(nodes[pos_vals]))