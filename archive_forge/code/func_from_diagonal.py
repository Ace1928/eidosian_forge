from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
@staticmethod
def from_diagonal(diagonal):
    """
        Return a representation of a covariance matrix from its diagonal.

        Parameters
        ----------
        diagonal : array_like
            The diagonal elements of a diagonal matrix.

        Notes
        -----
        Let the diagonal elements of a diagonal covariance matrix :math:`D` be
        stored in the vector :math:`d`.

        When all elements of :math:`d` are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x \\cdot d^{-1/2}`, where the inverse square root can be taken
        element-wise.
        :math:`\\log\\det{D}` is calculated as :math:`-2 \\sum(\\log{d})`,
        where the :math:`\\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive elements of :math:`d` are
        ignored. Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive elements of :math:`d` as zeros.

        Examples
        --------
        Prepare a symmetric positive definite covariance matrix ``A`` and a
        data point ``x``.

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> A = np.diag(rng.random(n))
        >>> x = rng.random(size=n)

        Extract the diagonal from ``A`` and create the `Covariance` object.

        >>> d = np.diag(A)
        >>> cov = stats.Covariance.from_diagonal(d)

        Compare the functionality of the `Covariance` object against a
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = np.diag(d**-0.5) @ x
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
    return CovViaDiagonal(diagonal)