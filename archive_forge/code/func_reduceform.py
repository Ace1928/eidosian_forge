import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def reduceform(self, apoly):
    """

        this assumes no exog, todo

        """
    if apoly.ndim != 3:
        raise ValueError('apoly needs to be 3d')
    nlags, nvarsex, nvars = apoly.shape
    a = np.empty_like(apoly)
    try:
        a0inv = np.linalg.inv(a[0, :nvars, :])
    except np.linalg.LinAlgError:
        raise ValueError('matrix not invertible', 'ask for implementation of pinv')
    for lag in range(nlags):
        a[lag] = np.dot(a0inv, apoly[lag])
    return a