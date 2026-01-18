import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def corr_thresholded(data, minabs=None, max_elt=10000000.0):
    """
    Construct a sparse matrix containing the thresholded row-wise
    correlation matrix from a data array.

    Parameters
    ----------
    data : array_like
        The data from which the row-wise thresholded correlation
        matrix is to be computed.
    minabs : non-negative real
        The threshold value; correlation coefficients smaller in
        magnitude than minabs are set to zero.  If None, defaults
        to 1 / sqrt(n), see Notes for more information.

    Returns
    -------
    cormat : sparse.coo_matrix
        The thresholded correlation matrix, in COO format.

    Notes
    -----
    This is an alternative to C = np.corrcoef(data); C \\*= (np.abs(C)
    >= absmin), suitable for very tall data matrices.

    If the data are jointly Gaussian, the marginal sampling
    distributions of the elements of the sample correlation matrix are
    approximately Gaussian with standard deviation 1 / sqrt(n).  The
    default value of ``minabs`` is thus equal to 1 standard error, which
    will set to zero approximately 68% of the estimated correlation
    coefficients for which the population value is zero.

    No intermediate matrix with more than ``max_elt`` values will be
    constructed.  However memory use could still be high if a large
    number of correlation values exceed `minabs` in magnitude.

    The thresholded matrix is returned in COO format, which can easily
    be converted to other sparse formats.

    Examples
    --------
    Here X is a tall data matrix (e.g. with 100,000 rows and 50
    columns).  The row-wise correlation matrix of X is calculated
    and stored in sparse form, with all entries smaller than 0.3
    treated as 0.

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> b = 1.5 - np.random.rand(10, 1)
    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)
    >>> cmat = corr_thresholded(x, 0.3)
    """
    nrow, ncol = data.shape
    if minabs is None:
        minabs = 1.0 / float(ncol)
    data = data.copy()
    data -= data.mean(1)[:, None]
    sd = data.std(1, ddof=1)
    ii = np.flatnonzero(sd > 1e-05)
    data[ii, :] /= sd[ii][:, None]
    ii = np.flatnonzero(sd <= 1e-05)
    data[ii, :] = 0
    bs = int(np.floor(max_elt / nrow))
    ipos_all, jpos_all, cor_values = ([], [], [])
    ir = 0
    while ir < nrow:
        ir2 = min(data.shape[0], ir + bs)
        cm = np.dot(data[ir:ir2, :], data.T) / (ncol - 1)
        cma = np.abs(cm)
        ipos, jpos = np.nonzero(cma >= minabs)
        ipos_all.append(ipos + ir)
        jpos_all.append(jpos)
        cor_values.append(cm[ipos, jpos])
        ir += bs
    ipos = np.concatenate(ipos_all)
    jpos = np.concatenate(jpos_all)
    cor_values = np.concatenate(cor_values)
    cmat = sparse.coo_matrix((cor_values, (ipos, jpos)), (nrow, nrow))
    return cmat