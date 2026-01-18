import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def stacksquare(self, a=None, name='ar', orientation='vertical'):
    """stack lagpolynomial vertically in 2d square array with eye

        """
    if a is not None:
        a = a
    elif name == 'ar':
        a = self.ar
    elif name == 'ma':
        a = self.ma
    else:
        raise ValueError('no array or name given')
    astacked = a.reshape(-1, self.nvarall)
    lenpk, nvars = astacked.shape
    amat = np.eye(lenpk, k=nvars)
    amat[:, :nvars] = astacked
    return amat