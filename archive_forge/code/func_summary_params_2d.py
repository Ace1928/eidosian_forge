from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def summary_params_2d(result, extras=None, endog_names=None, exog_names=None, title=None):
    """create summary table of regression parameters with several equations

    This allows interleaving of parameters with bse and/or tvalues

    Parameters
    ----------
    result : result instance
        the result instance with params and attributes in extras
    extras : list[str]
        additional attributes to add below a parameter row, e.g. bse or tvalues
    endog_names : {list[str], None}
        names for rows of the parameter array (multivariate endog)
    exog_names : {list[str], None}
        names for columns of the parameter array (exog)
    alpha : float
        level for confidence intervals, default 0.95
    title : None or string

    Returns
    -------
    tables : list of SimpleTable
        this contains a list of all seperate Subtables
    table_all : SimpleTable
        the merged table with results concatenated for each row of the parameter
        array

    """
    if endog_names is None:
        endog_names = ['endog_%d' % i for i in np.unique(result.model.endog)[1:]]
    if exog_names is None:
        exog_names = ['var%d' % i for i in range(len(result.params))]
    res_params = [[forg(item, prec=4) for item in row] for row in result.params]
    if extras:
        extras_list = [[['%10s' % ('(' + forg(v, prec=3).strip() + ')') for v in col] for col in getattr(result, what)] for what in extras]
        data = lzip(res_params, *extras_list)
        data = [i for j in data for i in j]
        stubs = lzip(endog_names, *[[''] * len(endog_names)] * len(extras))
        stubs = [i for j in stubs for i in j]
    else:
        data = res_params
        stubs = endog_names
    txt_fmt = copy.deepcopy(fmt_params)
    txt_fmt['data_fmts'] = ['%s'] * result.params.shape[1]
    return SimpleTable(data, headers=exog_names, stubs=stubs, title=title, txt_fmt=txt_fmt)