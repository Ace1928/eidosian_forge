import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def graph_multiresolution(G, levels, sparsify=True, sparsify_eps=None, downsampling_method='largest_eigenvector', reduction_method='kron', compute_full_eigen=False, reg_eps=0.005):
    """Compute a pyramid of graphs (by Kron reduction).

    'graph_multiresolution(G,levels)' computes a multiresolution of
    graph by repeatedly downsampling and performing graph reduction. The
    default downsampling method is the largest eigenvector method based on
    the polarity of the components of the eigenvector associated with the
    largest graph Laplacian eigenvalue. The default graph reduction method
    is Kron reduction followed by a graph sparsification step.
    *param* is a structure of optional parameters.

    Parameters
    ----------
    G : Graph structure
        The graph to reduce.
    levels : int
        Number of level of decomposition
    lambd : float
        Stability parameter. It adds self loop to the graph to give the
        algorithm some stability (default = 0.025). [UNUSED?!]
    sparsify : bool
        To perform a spectral sparsification step immediately after
        the graph reduction (default is True).
    sparsify_eps : float
        Parameter epsilon used in the spectral sparsification
        (default is min(10/sqrt(G.N),.3)).
    downsampling_method: string
        The graph downsampling method (default is 'largest_eigenvector').
    reduction_method : string
        The graph reduction method (default is 'kron')
    compute_full_eigen : bool
        To also compute the graph Laplacian eigenvalues and eigenvectors
        for every graph in the multiresolution sequence (default is False).
    reg_eps : float
        The regularized graph Laplacian is :math:`\\bar{L}=L+\\epsilon I`.
        A smaller epsilon may lead to better regularization, but will also
        require a higher order Chebyshev approximation. (default is 0.005)

    Returns
    -------
    Gs : list
        A list of graph layers.

    Examples
    --------
    >>> from pygsp import reduction
    >>> levels = 5
    >>> G = graphs.Sensor(N=512)
    >>> G.compute_fourier_basis()
    >>> Gs = reduction.graph_multiresolution(G, levels, sparsify=False)
    >>> for idx in range(levels):
    ...     Gs[idx].plotting['plot_name'] = 'Reduction level: {}'.format(idx)
    ...     Gs[idx].plot()

    """
    if sparsify_eps is None:
        sparsify_eps = min(10.0 / np.sqrt(G.N), 0.3)
    if compute_full_eigen:
        G.compute_fourier_basis()
    else:
        G.estimate_lmax()
    Gs = [G]
    Gs[0].mr = {'idx': np.arange(G.N), 'orig_idx': np.arange(G.N)}
    for i in range(levels):
        if downsampling_method == 'largest_eigenvector':
            if hasattr(Gs[i], '_U'):
                V = Gs[i].U[:, -1]
            else:
                V = linalg.eigs(Gs[i].L, 1)[1][:, 0]
            V *= np.sign(V[0])
            ind = np.nonzero(V >= 0)[0]
        else:
            raise NotImplementedError('Unknown graph downsampling method.')
        if reduction_method == 'kron':
            Gs.append(kron_reduction(Gs[i], ind))
        else:
            raise NotImplementedError('Unknown graph reduction method.')
        if sparsify and Gs[i + 1].N > 2:
            Gs[i + 1] = graph_sparsify(Gs[i + 1], min(max(sparsify_eps, 2.0 / np.sqrt(Gs[i + 1].N)), 1.0))
        if compute_full_eigen:
            Gs[i + 1].compute_fourier_basis()
        else:
            Gs[i + 1].estimate_lmax()
        Gs[i + 1].mr = {'idx': ind, 'orig_idx': Gs[i].mr['orig_idx'][ind], 'level': i}
        L_reg = Gs[i].L + reg_eps * sparse.eye(Gs[i].N)
        Gs[i].mr['K_reg'] = kron_reduction(L_reg, ind)
        Gs[i].mr['green_kernel'] = filters.Filter(Gs[i], lambda x: 1.0 / (reg_eps + x))
    return Gs