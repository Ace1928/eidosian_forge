import numpy as np
from scipy import stats
def se_vectorized(self):
    """standard error for each equation (row) treated separately

        """
    var = self.var()
    return np.sqrt(var)