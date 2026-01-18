import numpy as np
from . import tools
@property
def posterior_cov_inv_chol_sparse(self):
    """
        Sparse Cholesky factor of inverse posterior covariance matrix

        Notes
        -----
        This attribute holds in sparse diagonal banded storage the Cholesky
        factor of the inverse of the posterior covariance matrix. If we denote
        :math:`P = Var[\\alpha \\mid Y^n ]`, then the this attribute holds the
        lower Cholesky factor :math:`L`, defined from :math:`L L' = P^{-1}`.
        This attribute uses the sparse diagonal banded storage described in the
        documentation of, for example, the SciPy function
        `scipy.linalg.solveh_banded`.
        """
    if self._posterior_cov_inv_chol is None:
        self._posterior_cov_inv_chol = np.array(self._simulation_smoother.posterior_cov_inv_chol, copy=True)
    return self._posterior_cov_inv_chol