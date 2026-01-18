from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
@property
def isinvertible(self):
    """
        Arma process is invertible if MA roots are outside unit circle.

        Returns
        -------
        bool
             True if moving average roots are outside unit circle.
        """
    if np.all(np.abs(self.maroots) > 1):
        return True
    else:
        return False