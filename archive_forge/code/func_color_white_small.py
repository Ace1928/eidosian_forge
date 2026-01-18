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
def color_white_small(val):
    """
                    Takes a scalar and returns a string with
                    the css property `'color: white'` for small values, black otherwise.

                    takes threshold from outer scope
                    """
    color = 'white' if np.abs(val) < threshold else 'black'
    return 'color: %s' % color