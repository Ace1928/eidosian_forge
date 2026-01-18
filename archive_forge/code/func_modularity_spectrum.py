import networkx as nx
@nx._dispatch
def modularity_spectrum(G):
    """Returns eigenvalues of the modularity matrix of G.

    Parameters
    ----------
    G : Graph
       A NetworkX Graph or DiGraph

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    See Also
    --------
    modularity_matrix

    References
    ----------
    .. [1] M. E. J. Newman, "Modularity and community structure in networks",
       Proc. Natl. Acad. Sci. USA, vol. 103, pp. 8577-8582, 2006.
    """
    import scipy as sp
    if G.is_directed():
        return sp.linalg.eigvals(nx.directed_modularity_matrix(G))
    else:
        return sp.linalg.eigvals(nx.modularity_matrix(G))