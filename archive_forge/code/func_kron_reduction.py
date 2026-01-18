import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def kron_reduction(G, ind):
    """Compute the Kron reduction.

    This function perform the Kron reduction of the weight matrix in the
    graph *G*, with boundary nodes labeled by *ind*. This function will
    create a new graph with a weight matrix Wnew that contain only boundary
    nodes and is computed as the Schur complement of the original matrix
    with respect to the selected indices.

    Parameters
    ----------
    G : Graph or sparse matrix
        Graph structure or weight matrix
    ind : list
        indices of the nodes to keep

    Returns
    -------
    Gnew : Graph or sparse matrix
        New graph structure or weight matrix


    References
    ----------
    See :cite:`dorfler2013kron`

    """
    if isinstance(G, graphs.Graph):
        if G.lap_type != 'combinatorial':
            msg = 'Unknown reduction for {} Laplacian.'.format(G.lap_type)
            raise NotImplementedError(msg)
        if G.is_directed():
            msg = 'This method only work for undirected graphs.'
            raise NotImplementedError(msg)
        L = G.L
    else:
        L = G
    N = np.shape(L)[0]
    ind_comp = np.setdiff1d(np.arange(N, dtype=int), ind)
    L_red = L[np.ix_(ind, ind)]
    L_in_out = L[np.ix_(ind, ind_comp)]
    L_out_in = L[np.ix_(ind_comp, ind)].tocsc()
    L_comp = L[np.ix_(ind_comp, ind_comp)].tocsc()
    Lnew = L_red - L_in_out.dot(linalg.spsolve(L_comp, L_out_in))
    if np.abs(Lnew - Lnew.T).sum() < np.spacing(1) * np.abs(Lnew).sum():
        Lnew = (Lnew + Lnew.T) / 2.0
    if isinstance(G, graphs.Graph):
        Wnew = sparse.diags(Lnew.diagonal(), 0) - Lnew
        Snew = Lnew.diagonal() - np.ravel(Wnew.sum(0))
        if np.linalg.norm(Snew, 2) >= np.spacing(1000):
            Wnew = Wnew + sparse.diags(Snew, 0)
        Wnew = Wnew - Wnew.diagonal()
        coords = G.coords[ind, :] if len(G.coords.shape) else np.ndarray(None)
        Gnew = graphs.Graph(W=Wnew, coords=coords, lap_type=G.lap_type, plotting=G.plotting, gtype='Kron reduction')
    else:
        Gnew = Lnew
    return Gnew