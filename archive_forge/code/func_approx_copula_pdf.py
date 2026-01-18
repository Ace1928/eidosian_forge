import warnings
import numpy as np
from scipy import interpolate, stats
def approx_copula_pdf(copula, k_bins=10, force_uniform=True, use_pdf=False):
    """Histogram probabilities as approximation to a copula density.

    Parameters
    ----------
    copula : instance
        Instance of a copula class. Only the ``pdf`` method is used.
    k_bins : int
        Number of bins along each dimension in the approximating histogram.
    force_uniform : bool
        If true, then the pdf grid will be adjusted to have uniform margins
        using `nearest_matrix_margin`.
        If false, then no adjustment is done and the margins may not be exactly
        uniform.
    use_pdf : bool
        If false, then the grid cell probabilities will be computed from the
        copula cdf.
        If true, then the density, ``pdf``, is used and cell probabilities
        are approximated by averaging the pdf of the cell corners. This is
        only useful if the cdf is not available.

    Returns
    -------
    bin probabilites : ndarray
        Probability that random variable falls in given bin. This corresponds
        to a discrete distribution, and is not scaled to bin size to form a
        piecewise uniform, histogram density.
        Bin probabilities are a k-dim array with k_bins segments in each
        dimensionrows.

    Notes
    -----
    This function is intended for internal use and will be generalized in
    future. API will change.
    """
    k_dim = copula.k_dim
    k = k_bins + 1
    ks = tuple([k] * k_dim)
    if use_pdf:
        g = _Grid([k] * k_dim, eps=0.1 / k_bins)
        pdfg = copula.pdf(g.x_flat).reshape(*ks)
        pdfg *= 1 / k ** k_dim
        ag = average_grid(pdfg)
        if force_uniform:
            pdf_grid = nearest_matrix_margins(ag, maxiter=100, tol=1e-08)
        else:
            pdf_grid = ag / ag.sum()
    else:
        g = _Grid([k] * k_dim, eps=1e-06)
        cdfg = copula.cdf(g.x_flat).reshape(*ks)
        pdf_grid = cdf2prob_grid(cdfg, prepend=None)
        pdf_grid /= pdf_grid.sum()
    return pdf_grid