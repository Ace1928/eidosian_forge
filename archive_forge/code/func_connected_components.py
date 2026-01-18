import cupy
import cupyx.scipy.sparse
def connected_components(csgraph, directed=True, connection='weak', return_labels=True):
    """Analyzes the connected components of a sparse graph

    Args:
        csgraph (cupy.ndarray of cupyx.scipy.sparse.csr_matrix): The adjacency
            matrix representing connectivity among nodes.
        directed (bool): If ``True``, it operates on a directed graph. If
            ``False``, it operates on an undirected graph.
        connection (str): ``'weak'`` or ``'strong'``. For directed graphs, the
            type of connection to use. Nodes i and j are "strongly" connected
            only when a path exists both from i to j and from j to i.
            If ``directed`` is ``False``, this argument is ignored.
        return_labels (bool): If ``True``, it returns the labels for each of
            the connected components.

    Returns:
        tuple of int and cupy.ndarray, or int:
            If ``return_labels`` == ``True``, returns a tuple ``(n, labels)``,
            where ``n`` is the number of connected components and ``labels`` is
            labels of each connected components. Otherwise, returns ``n``.

    .. seealso:: :func:`scipy.sparse.csgraph.connected_components`
    """
    if not pylibcugraph_available:
        raise RuntimeError('pylibcugraph is not available')
    connection = connection.lower()
    if connection not in ('weak', 'strong'):
        raise ValueError("connection must be 'weak' or 'strong'")
    if not directed:
        connection = 'weak'
    if csgraph.ndim != 2:
        raise ValueError('graph should have two dimensions')
    if not cupyx.scipy.sparse.isspmatrix_csr(csgraph):
        csgraph = cupyx.scipy.sparse.csr_matrix(csgraph)
    m, m1 = csgraph.shape
    if m != m1:
        raise ValueError('graph should be a square array')
    if csgraph.nnz == 0:
        return (m, cupy.arange(m, dtype=csgraph.indices.dtype))
    labels = cupy.empty(m, dtype=csgraph.indices.dtype)
    if connection == 'strong':
        pylibcugraph.strongly_connected_components(offsets=csgraph.indptr, indices=csgraph.indices, weights=None, num_verts=m, num_edges=csgraph.nnz, labels=labels)
    else:
        csgraph += csgraph.T
        if not cupyx.scipy.sparse.isspmatrix_csr(csgraph):
            csgraph = cupyx.scipy.sparse.csr_matrix(csgraph)
        pylibcugraph.weakly_connected_components(offsets=csgraph.indptr, indices=csgraph.indices, weights=None, num_verts=m, num_edges=csgraph.nnz, labels=labels)
        labels -= 1
    count = cupy.zeros((1,), dtype=csgraph.indices.dtype)
    root_labels = cupy.empty((m,), dtype=csgraph.indices.dtype)
    _cupy_count_components(labels, count, root_labels, size=m)
    n = int(count[0])
    if not return_labels:
        return n
    _cupy_adjust_labels(n, cupy.sort(root_labels[:n]), labels)
    return (n, labels)