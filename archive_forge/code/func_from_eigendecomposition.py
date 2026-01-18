from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
@staticmethod
def from_eigendecomposition(eigendecomposition):
    """
        Representation of a covariance provided via eigendecomposition

        Parameters
        ----------
        eigendecomposition : sequence
            A sequence (nominally a tuple) containing the eigenvalue and
            eigenvector arrays as computed by `scipy.linalg.eigh` or
            `numpy.linalg.eigh`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, let :math:`V` be matrix of
        eigenvectors, and let :math:`W` be the diagonal matrix of eigenvalues
        such that `V W V^T = A`.

        When all of the eigenvalues are strictly positive, whitening of a
        data point :math:`x` is performed by computing
        :math:`x^T (V W^{-1/2})`, where the inverse square root can be taken
        element-wise.
        :math:`\\log\\det{A}` is calculated as  :math:`tr(\\log{W})`,
        where the :math:`\\log` operation is performed element-wise.

        This `Covariance` class supports singular covariance matrices. When
        computing ``_log_pdet``, non-positive eigenvalues are ignored.
        Whitening is not well defined when the point to be whitened
        does not lie in the span of the columns of the covariance matrix. The
        convention taken here is to treat the inverse square root of
        non-positive eigenvalues as zeros.

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

        Perform the eigendecomposition of ``A`` and create the `Covariance`
        object.

        >>> w, v = np.linalg.eigh(A)
        >>> cov = stats.Covariance.from_eigendecomposition((w, v))

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ (v @ np.diag(w**-0.5))
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = np.linalg.slogdet(A)[-1]
        >>> np.allclose(res, ref)
        True

        """
    return CovViaEigendecomposition(eigendecomposition)