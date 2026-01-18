import numpy as np
def psi_deriv(self, z):
    """
        The derivative of MQuantileNorm function

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi_deriv : ndarray

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
    qq = self._get_q(z)
    return qq * self.base_norm.psi_deriv(z)