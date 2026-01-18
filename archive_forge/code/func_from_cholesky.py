from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
@staticmethod
def from_cholesky(cholesky):
    """
        Representation of a covariance provided via the (lower) Cholesky factor

        Parameters
        ----------
        cholesky : array_like
            The lower triangular Cholesky factor of the covariance matrix.

        Notes
        -----
        Let the covariance matrix be :math:`A` and :math:`L` be the lower
        Cholesky factor such that :math:`L L^T = A`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`L^{-1} x`. :math:`\\log\\det{A}` is calculated as
        :math:`2tr(\\log{L})`, where the :math:`\\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the Cholesky decomposition does not exist for a singular
        covariance matrix.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = rng.random(size=(n, n))
        >>> A = A @ A.T  # make the covariance symmetric positive definite
        >>> x = rng.random(size=n)

        Perform the Cholesky decomposition of ``A`` and create the
        `Covariance` object.

        >>> L = np.linalg.cholesky(A)
        >>> cov = stats.Covariance.from_cholesky(L)

        Compare the functionality of the `Covariance` object against
        reference implementation.

        >>> from scipy.linalg import solve_triangular
        >>> res = cov.whiten(x)
        >>> ref = solve_triangular(L, x, lower=True)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
    return CovViaCholesky(cholesky)