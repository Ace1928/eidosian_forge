import numpy as np
from . import tools
@property
def posterior_cov(self):
    """
        Posterior covariance of the states conditional on the data

        Notes
        -----
        **Warning**: the matrix computed when accessing this property can be
        extremely large: it is shaped `(nobs * k_states, nobs * k_states)`. In
        most cases, it is better to use the `posterior_cov_inv_chol_sparse`
        property if possible, which holds in sparse diagonal banded storage
        the Cholesky factor of the inverse of the posterior covariance matrix.

        .. math::

            Var[\\alpha \\mid Y^n ]

        This posterior covariance matrix is *not* identical to the
        `smoothed_state_cov` attribute produced by the Kalman smoother, because
        it additionally contains all cross-covariance terms. Instead,
        `smoothed_state_cov` contains the `(k_states, k_states)` block
        diagonal entries of this posterior covariance matrix.
        """
    if self._posterior_cov is None:
        from scipy.linalg import cho_solve_banded
        inv_chol = self.posterior_cov_inv_chol_sparse
        self._posterior_cov = cho_solve_banded((inv_chol, True), np.eye(inv_chol.shape[1]))
    return self._posterior_cov