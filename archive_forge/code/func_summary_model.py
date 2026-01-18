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
def summary_model(results):
    """
    Create a dict with information about the model
    """

    def time_now(*args, **kwds):
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d %H:%M')
    info = {}
    info['Model:'] = lambda x: x.model.__class__.__name__
    info['Model Family:'] = lambda x: x.family.__class.__name__
    info['Link Function:'] = lambda x: x.family.link.__class__.__name__
    info['Dependent Variable:'] = lambda x: x.model.endog_names
    info['Date:'] = time_now
    info['No. Observations:'] = lambda x: '%#6d' % x.nobs
    info['Df Model:'] = lambda x: '%#6d' % x.df_model
    info['Df Residuals:'] = lambda x: '%#6d' % x.df_resid
    info['Converged:'] = lambda x: x.mle_retvals['converged']
    info['No. Iterations:'] = lambda x: x.mle_retvals['iterations']
    info['Method:'] = lambda x: x.method
    info['Norm:'] = lambda x: x.fit_options['norm']
    info['Scale Est.:'] = lambda x: x.fit_options['scale_est']
    info['Cov. Type:'] = lambda x: x.fit_options['cov']
    rsquared_type = '' if results.k_constant else ' (uncentered)'
    info['R-squared' + rsquared_type + ':'] = lambda x: '%#8.3f' % x.rsquared
    info['Adj. R-squared' + rsquared_type + ':'] = lambda x: '%#8.3f' % x.rsquared_adj
    info['Pseudo R-squared:'] = lambda x: '%#8.3f' % x.prsquared
    info['AIC:'] = lambda x: '%8.4f' % x.aic
    info['BIC:'] = lambda x: '%8.4f' % x.bic
    info['Log-Likelihood:'] = lambda x: '%#8.5g' % x.llf
    info['LL-Null:'] = lambda x: '%#8.5g' % x.llnull
    info['LLR p-value:'] = lambda x: '%#8.5g' % x.llr_pvalue
    info['Deviance:'] = lambda x: '%#8.5g' % x.deviance
    info['Pearson chi2:'] = lambda x: '%#6.3g' % x.pearson_chi2
    info['F-statistic:'] = lambda x: '%#8.4g' % x.fvalue
    info['Prob (F-statistic):'] = lambda x: '%#6.3g' % x.f_pvalue
    info['Scale:'] = lambda x: '%#8.5g' % x.scale
    out = {}
    for key, func in info.items():
        try:
            out[key] = func(results)
        except (AttributeError, KeyError, NotImplementedError):
            pass
    return out