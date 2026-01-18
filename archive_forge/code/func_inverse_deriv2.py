import numpy as np
import scipy.stats
import warnings
def inverse_deriv2(self, z):
    """
        Second derivative of the inverse of the Log-Log transform link function

        Parameters
        ----------
        z : array_like
            The value of the inverse of the LogLog link function at `p`

        Returns
        -------
        g^(-1)''(z) : ndarray
            The second derivative of the inverse of the LogLog link function
        """
    return self.inverse_deriv(z) * (np.exp(-z) - 1)