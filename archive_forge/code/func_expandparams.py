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
def expandparams(self, params):
    """
        expand to full parameter array when some parameters are fixed

        Parameters
        ----------
        params : ndarray
            reduced parameter array

        Returns
        -------
        paramsfull : ndarray
            expanded parameter array where fixed parameters are included

        Notes
        -----
        Calling this requires that self.fixed_params and self.fixed_paramsmask
        are defined.

        *developer notes:*

        This can be used in the log-likelihood to ...

        this could also be replaced by a more general parameter
        transformation.
        """
    paramsfull = self.fixed_params.copy()
    paramsfull[self.fixed_paramsmask] = params
    return paramsfull