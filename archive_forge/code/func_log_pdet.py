from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
@property
def log_pdet(self):
    """
        Log of the pseudo-determinant of the covariance matrix
        """
    return np.array(self._log_pdet, dtype=float)[()]