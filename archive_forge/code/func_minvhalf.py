import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def minvhalf(self):
    evals, evecs = self.meigh
    return np.dot(evecs, 1.0 / np.sqrt(evals) * evecs.T)