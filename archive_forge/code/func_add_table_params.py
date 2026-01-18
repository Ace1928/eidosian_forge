from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def add_table_params(self, res, yname=None, xname=None, alpha=0.05, use_t=True):
    """create and add a table for the parameter estimates

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        yname : {str, None}
            optional name for the endogenous variable, default is "y"
        xname : {list[str], None}
            optional names for the exogenous variables, default is "var_xx"
        alpha : float
            significance level for the confidence intervals
        use_t : bool
            indicator whether the p-values are based on the Student-t
            distribution (if True) or on the normal distribution (if False)

        Returns
        -------
        None : table is attached

        """
    if res.params.ndim == 1:
        table = summary_params(res, yname=yname, xname=xname, alpha=alpha, use_t=use_t)
    elif res.params.ndim == 2:
        _, table = summary_params_2dflat(res, endog_names=yname, exog_names=xname, alpha=alpha, use_t=use_t)
    else:
        raise ValueError('params has to be 1d or 2d')
    self.tables.append(table)