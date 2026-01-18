import numpy as np
from scipy import integrate, stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.validation import array_like, float_like
from . import bandwidths
from .kdetools import forrt, revrt, silverman_transform
from .linbin import fast_linbin
def kdensityfft(x, kernel='gau', bw='normal_reference', weights=None, gridsize=None, adjust=1, clip=(-np.inf, np.inf), cut=3, retgrid=True):
    """
    Rosenblatt-Parzen univariate kernel density estimator

    Parameters
    ----------
    x : array_like
        The variable for which the density estimate is desired.
    kernel : str
        ONLY GAUSSIAN IS CURRENTLY IMPLEMENTED.
        "bi" for biweight
        "cos" for cosine
        "epa" for Epanechnikov, default
        "epa2" for alternative Epanechnikov
        "gau" for Gaussian.
        "par" for Parzen
        "rect" for rectangular
        "tri" for triangular
    bw : str, float, callable
        The bandwidth to use. Choices are:

        - "scott" - 1.059 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "silverman" - .9 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "normal_reference" - C * A * nobs ** (-1/5.), where C is
          calculated from the kernel. Equivalent (up to 2 dp) to the
          "scott" bandwidth for gaussian kernels. See bandwidths.py
        - If a float is given, its value is used as the bandwidth.
        - If a callable is given, it's return value is used.
          The callable should take exactly two parameters, i.e.,
          fn(x, kern), and return a float, where:

          * x - the clipped input data
          * kern - the kernel instance used

    weights : array or None
        WEIGHTS ARE NOT CURRENTLY IMPLEMENTED.
        Optional  weights. If the x value is clipped, then this weight is
        also dropped.
    gridsize : int
        If gridsize is None, min(len(x), 512) is used. Note that the provided
        number is rounded up to the next highest power of 2.
    adjust : float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
        clip : tuple
        Observations in x that are outside of the range given by clip are
        dropped. The number of observations in x is then shortened.
    cut : float
        Defines the length of the grid past the lowest and highest values of x
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{x.min() or x.max()}
    retgrid : bool
        Whether or not to return the grid over which the density is estimated.

    Returns
    -------
    density : ndarray
        The densities estimated at the grid points.
    grid : ndarray, optional
        The grid points at which the density is estimated.

    Notes
    -----
    Only the default kernel is implemented. Weights are not implemented yet.
    This follows Silverman (1982) with changes suggested by Jones and Lotwick
    (1984). However, the discretization step is replaced by linear binning
    of Fan and Marron (1994). This should be extended to accept the parts
    that are dependent only on the data to speed things up for
    cross-validation.

    References
    ----------
    Fan, J. and J.S. Marron. (1994) `Fast implementations of nonparametric
        curve estimators`. Journal of Computational and Graphical Statistics.
        3.1, 35-56.
    Jones, M.C. and H.W. Lotwick. (1984) `Remark AS R50: A Remark on Algorithm
        AS 176. Kernal Density Estimation Using the Fast Fourier Transform`.
        Journal of the Royal Statistical Society. Series C. 33.1, 120-2.
    Silverman, B.W. (1982) `Algorithm AS 176. Kernel density estimation using
        the Fast Fourier Transform. Journal of the Royal Statistical Society.
        Series C. 31.2, 93-9.
    """
    x = np.asarray(x)
    x = x[np.logical_and(x > clip[0], x < clip[1])]
    kern = kernel_switch[kernel]()
    if callable(bw):
        bw = float(bw(x, kern))
    elif isinstance(bw, str):
        bw = bandwidths.select_bandwidth(x, bw, kern)
    else:
        bw = float_like(bw, 'bw')
    bw *= adjust
    nobs = len(x)
    if gridsize is None:
        gridsize = np.max((nobs, 512.0))
    gridsize = 2 ** np.ceil(np.log2(gridsize))
    a = np.min(x) - cut * bw
    b = np.max(x) + cut * bw
    grid, delta = np.linspace(a, b, int(gridsize), retstep=True)
    RANGE = b - a
    binned = fast_linbin(x, a, b, gridsize) / (delta * nobs)
    y = forrt(binned)
    zstar = silverman_transform(bw, gridsize, RANGE) * y
    f = revrt(zstar)
    if retgrid:
        return (f, grid, bw)
    else:
        return (f, bw)