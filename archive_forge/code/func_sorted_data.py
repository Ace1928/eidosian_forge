from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
@cache_readonly
def sorted_data(self):
    """sorted data"""
    sorted_data = np.array(self.data, copy=True)
    sorted_data.sort()
    return sorted_data