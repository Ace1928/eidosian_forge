import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def minv_y(self, y):
    return np.dot(self.minv, y)