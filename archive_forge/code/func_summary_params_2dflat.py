from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def summary_params_2dflat(result, endog_names=None, exog_names=None, alpha=0.05, use_t=True, keep_headers=True, endog_cols=False):
    """summary table for parameters that are 2d, e.g. multi-equation models

    Parameters
    ----------
    result : result instance
        the result instance with params, bse, tvalues and conf_int
    endog_names : {list[str], None}
        names for rows of the parameter array (multivariate endog)
    exog_names : {list[str], None}
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    keep_headers : bool
        If true (default), then sub-tables keep their headers. If false, then
        only the first headers are kept, the other headerse are blanked out
    endog_cols : bool
        If false (default) then params and other result statistics have
        equations by rows. If true, then equations are assumed to be in columns.
        Not implemented yet.

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the parameter
        array

    """
    res = result
    params = res.params
    if params.ndim == 2:
        n_equ = params.shape[1]
        if len(endog_names) != params.shape[1]:
            raise ValueError('endog_names has wrong length')
    else:
        if len(endog_names) != len(params):
            raise ValueError('endog_names has wrong length')
        n_equ = 1
    if not isinstance(endog_names, list):
        if endog_names is None:
            endog_basename = 'endog'
        else:
            endog_basename = endog_names
        endog_names = res.model.endog_names[1:]
    tables = []
    for eq in range(n_equ):
        restup = (res, res.params[:, eq], res.bse[:, eq], res.tvalues[:, eq], res.pvalues[:, eq], res.conf_int(alpha)[eq])
        skiph = False
        tble = summary_params(restup, yname=endog_names[eq], xname=exog_names, alpha=alpha, use_t=use_t, skip_header=skiph)
        tables.append(tble)
    for i in range(len(endog_names)):
        tables[i].title = endog_names[i]
    table_all = table_extend(tables, keep_headers=keep_headers)
    return (tables, table_all)