import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
@staticmethod
def str2matrix(A):
    A = A.lstrip().rstrip().split('\n')
    A = np.array([row.split() for row in A]).astype(float)
    return A