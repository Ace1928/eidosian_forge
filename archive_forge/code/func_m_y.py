import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def m_y(self, y):
    return np.dot(self.m, y)