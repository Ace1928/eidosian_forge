import numpy as np
from scipy.optimize import fmin_ncg
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.optimize import _newton_cg
def grad_hess(x):
    return (grad(x), lambda x: A.T.dot(A.dot(x)))