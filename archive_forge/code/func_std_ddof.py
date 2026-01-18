import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def std_ddof(self, ddof=0):
    """standard deviation of data with given ddof

        Parameters
        ----------
        ddof : int, float
            degrees of freedom correction, independent of attribute ddof

        Returns
        -------
        std : float, ndarray
            standard deviation with denominator ``sum_weights - ddof``
        """
    return np.sqrt(self.var_ddof(ddof=ddof))