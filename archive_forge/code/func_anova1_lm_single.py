from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def anova1_lm_single(model, endog, exog, nobs, design_info, table, n_rows, test, pr_test, robust):
    """
    Anova table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    effects = getattr(model, 'effects', None)
    if effects is None:
        q, r = np.linalg.qr(exog)
        effects = np.dot(q.T, endog)
    arr = np.zeros((len(design_info.terms), len(design_info.column_names)))
    slices = [design_info.slice(name) for name in design_info.term_names]
    for i, slice_ in enumerate(slices):
        arr[i, slice_] = 1
    sum_sq = np.dot(arr, effects ** 2)
    idx = _intercept_idx(design_info)
    sum_sq = sum_sq[~idx]
    term_names = np.array(design_info.term_names)
    term_names = term_names[~idx]
    index = term_names.tolist()
    table.index = Index(index + ['Residual'])
    table.loc[index, ['df', 'sum_sq']] = np.c_[arr[~idx].sum(1), sum_sq]
    table.loc['Residual', ['sum_sq', 'df']] = (model.ssr, model.df_resid)
    if test == 'F':
        table[test] = table['sum_sq'] / table['df'] / (model.ssr / model.df_resid)
        table[pr_test] = stats.f.sf(table['F'], table['df'], model.df_resid)
        table.loc['Residual', [test, pr_test]] = (np.nan, np.nan)
    table['mean_sq'] = table['sum_sq'] / table['df']
    return table