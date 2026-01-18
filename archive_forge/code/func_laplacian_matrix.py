import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def laplacian_matrix(G, nodelist=None, weight='weight'):
    """Returns the Laplacian matrix of G.

    The graph Laplacian is the matrix L = D - A, where
    A is the adjacency matrix and D is the diagonal matrix of node degrees.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    L : SciPy sparse array
      The Laplacian matrix of G.

    Notes
    -----
    For MultiGraph, the edges weights are summed.

    See Also
    --------
    :func:`~networkx.convert_matrix.to_numpy_array`
    normalized_laplacian_matrix
    :func:`~networkx.linalg.spectrum.laplacian_spectrum`

    Examples
    --------
    For graphs with multiple connected components, L is permutation-similar
    to a block diagonal matrix where each block is the respective Laplacian
    matrix for each component.

    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])
    >>> print(nx.laplacian_matrix(G).toarray())
    [[ 1 -1  0  0  0]
     [-1  2 -1  0  0]
     [ 0 -1  1  0  0]
     [ 0  0  0  1 -1]
     [ 0  0  0 -1  1]]

    """
    import scipy as sp
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format='csr')
    n, m = A.shape
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format='csr'))
    return D - A