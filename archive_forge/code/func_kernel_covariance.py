import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def kernel_covariance(exog, loc, groups, kernel=None, bw=None):
    """
    Use kernel averaging to estimate a multivariate covariance function.

    The goal is to estimate a covariance function C(x, y) =
    cov(Z(x), Z(y)) where x, y are vectors in R^p (e.g. representing
    locations in time or space), and Z(.) represents a multivariate
    process on R^p.

    The data used for estimation can be observed at arbitrary values of the
    position vector, and there can be multiple independent observations
    from the process.

    Parameters
    ----------
    exog : array_like
        The rows of exog are realizations of the process obtained at
        specified points.
    loc : array_like
        The rows of loc are the locations (e.g. in space or time) at
        which the rows of exog are observed.
    groups : array_like
        The values of groups are labels for distinct independent copies
        of the process.
    kernel : MultivariateKernel instance, optional
        An instance of MultivariateKernel, defaults to
        GaussianMultivariateKernel.
    bw : array_like or scalar
        A bandwidth vector, or bandwidth multiplier.  If a 1d array, it
        contains kernel bandwidths for each component of the process, and
        must have length equal to the number of columns of exog.  If a scalar,
        bw is a bandwidth multiplier used to adjust the default bandwidth; if
        None, a default bandwidth is used.

    Returns
    -------
    A real-valued function C(x, y) that returns an estimate of the covariance
    between values of the process located at x and y.

    References
    ----------
    .. [1] Genton M, W Kleiber (2015).  Cross covariance functions for
        multivariate geostatics.  Statistical Science 30(2).
        https://arxiv.org/pdf/1507.08017.pdf
    """
    exog = np.asarray(exog)
    loc = np.asarray(loc)
    groups = np.asarray(groups)
    if loc.ndim == 1:
        loc = loc[:, None]
    v = [exog.shape[0], loc.shape[0], len(groups)]
    if min(v) != max(v):
        msg = 'exog, loc, and groups must have the same number of rows'
        raise ValueError(msg)
    ix = {}
    for i, g in enumerate(groups):
        if g not in ix:
            ix[g] = []
        ix[g].append(i)
    for g in ix.keys():
        ix[g] = np.sort(ix[g])
    if kernel is None:
        kernel = GaussianMultivariateKernel()
    if bw is None:
        kernel.set_default_bw(loc)
    elif np.isscalar(bw):
        kernel.set_default_bw(loc, bwm=bw)
    else:
        kernel.set_bandwidth(bw)

    def cov(x, y):
        kx = kernel.call(x, loc)
        ky = kernel.call(y, loc)
        cm, cw = (0.0, 0.0)
        for g, ii in ix.items():
            m = len(ii)
            j1, j2 = np.indices((m, m))
            j1 = ii[j1.flat]
            j2 = ii[j2.flat]
            w = kx[j1] * ky[j2]
            cm += np.einsum('ij,ik,i->jk', exog[j1, :], exog[j2, :], w)
            cw += w.sum()
        if cw < 1e-10:
            msg = 'Effective sample size is 0.  The bandwidth may be too ' + 'small, or you are outside the range of your data.'
            warnings.warn(msg)
            return np.nan * np.ones_like(cm)
        return cm / cw
    return cov