import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def minv(self):
    sinvv = np.dot(self.sinvdiag, self.v)
    return np.dot(sinvv.T, sinvv)