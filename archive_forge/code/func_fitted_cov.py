import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
@cache_readonly
def fitted_cov(self):
    """
        Returns the fitted covariance matrix.
        """
    c = np.dot(self.loadings, self.loadings.T)
    c.flat[::c.shape[0] + 1] += self.uniqueness
    return c