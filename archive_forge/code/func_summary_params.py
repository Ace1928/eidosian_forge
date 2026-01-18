from statsmodels.compat.pandas import FUTURE_STACK
from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt
def summary_params(results, yname=None, xname=None, alpha=0.05, use_t=True, skip_header=False, float_format='%.4f'):
    """create a summary table of parameters from results instance

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
    skip_header : bool
        If false (default), then the header row is added. If true, then no
        header row is added.
    float_format : str
        float formatting options (e.g. ".3g")

    Returns
    -------
    params_table : SimpleTable instance
    """
    if isinstance(results, tuple):
        results, params, bse, tvalues, pvalues, conf_int = results
    else:
        params = results.params
        bse = results.bse
        tvalues = results.tvalues
        pvalues = results.pvalues
        conf_int = results.conf_int(alpha)
    data = np.array([params, bse, tvalues, pvalues]).T
    data = np.hstack([data, conf_int])
    data = pd.DataFrame(data)
    if use_t:
        data.columns = ['Coef.', 'Std.Err.', 't', 'P>|t|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
    else:
        data.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
    if not xname:
        try:
            data.index = results.model.data.param_names
        except AttributeError:
            data.index = results.model.exog_names
    else:
        data.index = xname
    return data