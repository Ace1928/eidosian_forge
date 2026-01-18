import networkx as nx
@nx._dispatch(edge_attrs='weight')
def adjacency_matrix(G, nodelist=None, dtype=None, weight='weight'):
    """Returns adjacency matrix of G.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    dtype : NumPy data-type, optional
        The desired data-type for the array.
        If None, then the NumPy default is used.

    weight : string or None, optional (default='weight')
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    A : SciPy sparse array
      Adjacency matrix representation of G.

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    If you want a pure Python adjacency matrix representation try
    networkx.convert.to_dict_of_dicts which will return a
    dictionary-of-dictionaries format that can be addressed as a
    sparse matrix.

    For MultiGraph/MultiDiGraph with parallel edges the weights are summed.
    See `to_numpy_array` for other options.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the edge weight attribute
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting SciPy sparse array can be modified as follows:

    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.adjacency_matrix(G)
    >>> print(A.todense())
    [[1]]
    >>> A.setdiag(A.diagonal() * 2)
    >>> print(A.todense())
    [[2]]

    See Also
    --------
    to_numpy_array
    to_scipy_sparse_array
    to_dict_of_dicts
    adjacency_spectrum
    """
    return nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)