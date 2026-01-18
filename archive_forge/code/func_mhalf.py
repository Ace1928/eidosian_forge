import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def mhalf(self):
    return np.dot(np.diag(self.s), self.v)