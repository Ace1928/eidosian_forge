from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def anova_lm(*args, **kwargs):
    """
    Anova table for one or more fitted linear models.

    Parameters
    ----------
    args : fitted linear model results instance
        One or more fitted linear models
    scale : float
        Estimate of variance, If None, will be estimated from the largest
        model. Default is None.
    test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".
    typ : str or int {"I","II","III"} or {1,2,3}
        The type of Anova test to perform. See notes.
    robust : {None, "hc0", "hc1", "hc2", "hc3"}
        Use heteroscedasticity-corrected coefficient covariance matrix.
        If robust covariance is desired, it is recommended to use `hc3`.

    Returns
    -------
    anova : DataFrame
        When args is a single model, return is DataFrame with columns:

        sum_sq : float64
            Sum of squares for model terms.
        df : float64
            Degrees of freedom for model terms.
        F : float64
            F statistic value for significance of adding model terms.
        PR(>F) : float64
            P-value for significance of adding model terms.

        When args is multiple models, return is DataFrame with columns:

        df_resid : float64
            Degrees of freedom of residuals in models.
        ssr : float64
            Sum of squares of residuals in models.
        df_diff : float64
            Degrees of freedom difference from previous model in args
        ss_dff : float64
            Difference in ssr from previous model in args
        F : float64
            F statistic comparing to previous model in args
        PR(>F): float64
            P-value for significance comparing to previous model in args

    Notes
    -----
    Model statistics are given in the order of args. Models must have been fit
    using the formula api.

    See Also
    --------
    model_results.compare_f_test, model_results.compare_lm_test

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.formula.api import ols
    >>> moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load
    >>> data = moore.data
    >>> data = data.rename(columns={"partner.status" :
    ...                             "partner_status"}) # make name pythonic
    >>> moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
    ...                 data=data).fit()
    >>> table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame
    >>> print(table)
    """
    typ = kwargs.get('typ', 1)
    if len(args) == 1:
        model = args[0]
        return anova_single(model, **kwargs)
    if typ not in [1, 'I']:
        raise ValueError('Multiple models only supported for type I. Got type %s' % str(typ))
    test = kwargs.get('test', 'F')
    scale = kwargs.get('scale', None)
    n_models = len(args)
    pr_test = 'Pr(>%s)' % test
    names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
    table = DataFrame(np.zeros((n_models, 6)), columns=names)
    if not scale:
        scale = args[-1].scale
    table['ssr'] = [mdl.ssr for mdl in args]
    table['df_resid'] = [mdl.df_resid for mdl in args]
    table.loc[table.index[1:], 'df_diff'] = -np.diff(table['df_resid'].values)
    table['ss_diff'] = -table['ssr'].diff()
    if test == 'F':
        table['F'] = table['ss_diff'] / table['df_diff'] / scale
        table[pr_test] = stats.f.sf(table['F'], table['df_diff'], table['df_resid'])
        table.loc[table['F'].isnull(), pr_test] = np.nan
    return table