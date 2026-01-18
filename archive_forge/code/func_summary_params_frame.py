from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def summary_params_frame(results, yname=None, xname=None, alpha=0.05, use_t=True):
    """
    Create a summary table for the parameters

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
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    """
    if isinstance(results, tuple):
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:
        params = results.params
        std_err = results.bse
        tvalues = results.tvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|', 'Conf. Int. Low', 'Conf. Int. Upp.']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|', 'Conf. Int. Low', 'Conf. Int. Upp.']
    _, xname = _getnames(results, yname=yname, xname=xname)
    from pandas import DataFrame
    table = np.column_stack((params, std_err, tvalues, pvalues, conf_int))
    return DataFrame(table, columns=param_header, index=xname)