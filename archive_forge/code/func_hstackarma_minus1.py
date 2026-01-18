import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def hstackarma_minus1(self):
    """stack ar and lagpolynomial vertically in 2d array

        this is the Kalman Filter representation, I think
        """
    a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
    return a.swapaxes(1, 2).reshape(-1, self.nvarall)