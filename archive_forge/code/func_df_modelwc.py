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
@cache_readonly
def df_modelwc(self):
    """Model WC"""
    k_extra = getattr(self.model, 'k_extra', 0)
    if hasattr(self, 'df_model'):
        if hasattr(self, 'k_constant'):
            hasconst = self.k_constant
        elif hasattr(self, 'hasconst'):
            hasconst = self.hasconst
        else:
            hasconst = 1
        return self.df_model + hasconst + k_extra
    else:
        return self.params.size