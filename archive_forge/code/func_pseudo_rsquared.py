from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def pseudo_rsquared(self, kind='mcf'):
    """
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
        """
    kind = kind.lower()
    if kind.startswith('mcf'):
        prsq = 1 - self.llf / self.llnull
    elif kind.startswith('cox') or kind in ['cs', 'lr']:
        prsq = 1 - np.exp((self.llnull - self.llf) * (2 / self.nobs))
    else:
        raise ValueError('only McFadden and Cox-Snell are available')
    return prsq