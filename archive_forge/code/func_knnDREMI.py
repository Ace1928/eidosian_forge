from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def knnDREMI(x, y, k=10, n_bins=20, n_mesh=3, n_jobs=1, plot=False, return_drevi=False, **kwargs):
    """Compute kNN conditional Density Resampled Estimate of Mutual Information.

    Calculates k-Nearest Neighbor conditional Density Resampled Estimate of
    Mutual Information as defined in Van Dijk et al, 2018. [1]_

    kNN-DREMI is an adaptation of DREMI (Krishnaswamy et al. 2014, [2]_) for
    single cell RNA-sequencing data. DREMI captures the functional relationship
    between two genes across their entire dynamic range. The key change to
    kNN-DREMI is the replacement of the heat diffusion-based kernel-density
    estimator from Botev et al., 2010 [3]_ by a k-nearest neighbor-based
    density estimator (Sricharan et al., 2012 [4]_), which has been shown to be
    an effective method for sparse and high dimensional datasets.

    Note that kNN-DREMI, like Mutual Information and DREMI, is not symmetric.
    Here we are estimating I(Y|X).

    Parameters
    ----------
    x : array-like, shape=[n_samples]
        Input data (independent feature)
    y : array-like, shape=[n_samples]
        Input data (dependent feature)
    k : int, range=[0:n_samples), optional (default: 10)
        Number of neighbors
    n_bins : int, range=[0:inf), optional (default: 20)
        Number of bins for density resampling
    n_mesh : int, range=[0:inf), optional (default: 3)
        In each bin, density will be calculcated around (mesh ** 2) points
    n_jobs : int, optional (default: 1)
        Number of threads used for kNN calculation
    plot : bool, optional (default: False)
        If True, DREMI create plots of the data like those seen in
        Fig 5C/D of van Dijk et al. 2018. (doi:10.1016/j.cell.2018.05.061).
    return_drevi : bool, optional (default: False)
        If True, return the DREVI normalized density matrix in addition
        to the DREMI score.
    **kwargs : additional arguments for `scprep.stats.plot_knnDREMI`

    Returns
    -------
    dremi : float
        kNN condtional Density resampled estimate of mutual information
    drevi : np.ndarray
        DREVI normalized density matrix. Only returned if `return_drevi`
        is True.

    Examples
    --------
    >>> import scprep
    >>> data = scprep.io.load_csv("my_data.csv")
    >>> dremi = scprep.stats.knnDREMI(data['GENE1'], data['GENE2'],
    ...                               plot=True,
    ...                               filename='dremi.png')

    References
    ----------
    .. [1] van Dijk D *et al.* (2018),
        *Recovering Gene Interactions from Single-Cell Data Using Data
        Diffusion*, `Cell <https://doi.org/10.1016/j.cell.2018.05.061>`_.
    .. [2] Krishnaswamy S  *et al.* (2014),
        *Conditional density-based analysis of T cell signaling in single-cell
        data*, `Science <https://doi.org/10.1126/science.1250689>`_.
    .. [3] Botev ZI *et al*. (2010), *Kernel density estimation via diffusion*,
        `The Annals of Statistics <https://doi.org/10.1214/10-AOS799>`_.
    .. [4] Sricharan K *et al*. (2012), *Estimation of nonlinear functionals of
        densities with confidence*, `IEEE Transactions on Information Theory
        <https://doi.org/10.1109/TIT.2012.2195549>`_.
    """
    x, y = _vector_coerce_two_dense(x, y)
    if np.count_nonzero(x - x[0]) == 0 or np.count_nonzero(y - y[0]) == 0:
        warnings.warn('Attempting to calculate kNN-DREMI on a constant array. Returning `0`', UserWarning)
        if return_drevi:
            return (0, np.zeros((n_bins, n_bins), dtype=float))
        else:
            return 0
    if not isinstance(k, numbers.Integral):
        raise ValueError('Expected k as an integer. Got {}'.format(type(k)))
    if not isinstance(n_bins, numbers.Integral):
        raise ValueError('Expected n_bins as an integer. Got {}'.format(type(n_bins)))
    if not isinstance(n_mesh, numbers.Integral):
        raise ValueError('Expected n_mesh as an integer. Got {}'.format(type(n_mesh)))
    x = stats.zscore(x)
    y = stats.zscore(y)
    x_bins = np.linspace(min(x), max(x), n_bins + 1)
    y_bins = np.linspace(min(y), max(y), n_bins + 1)
    x_mesh = np.linspace(min(x), max(x), (n_mesh + 1) * n_bins + 1)
    y_mesh = np.linspace(min(y), max(y), (n_mesh + 1) * n_bins + 1)
    mesh_points = np.vstack([np.tile(x_mesh, len(y_mesh)), np.repeat(y_mesh, len(x_mesh))]).T
    knn = neighbors.NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(np.vstack([x, y]).T)
    dists, _ = knn.kneighbors(mesh_points)
    area = np.pi * dists[:, -1] ** 2
    density = k / area
    mesh_mask = np.logical_or(np.isin(mesh_points[:, 0], x_bins), np.isin(mesh_points[:, 1], y_bins))
    bin_density, _, _ = np.histogram2d(mesh_points[~mesh_mask, 0], mesh_points[~mesh_mask, 1], bins=[x_bins, y_bins], weights=density[~mesh_mask])
    bin_density = bin_density.T
    bin_density = bin_density / np.sum(bin_density)
    drevi = bin_density / np.sum(bin_density, axis=0)
    cond_entropies = stats.entropy(drevi, base=2)
    marginal_entropy = stats.entropy(np.sum(bin_density, axis=1), base=2)
    cond_sums = np.sum(bin_density, axis=0)
    conditional_entropy = np.sum(cond_entropies * cond_sums)
    mutual_info = marginal_entropy - conditional_entropy
    marginal_entropy_norm = stats.entropy(np.sum(drevi, axis=1), base=2)
    cond_sums_norm = np.mean(drevi)
    conditional_entropy_norm = np.sum(cond_entropies * cond_sums_norm)
    dremi = marginal_entropy_norm - conditional_entropy_norm
    if plot:
        plot_knnDREMI(dremi, mutual_info, x, y, n_bins, n_mesh, density, bin_density, drevi, **kwargs)
    if return_drevi:
        return (dremi, drevi)
    else:
        return dremi