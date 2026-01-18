import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def outlier_test(model_results, method='bonf', alpha=0.05, labels=None, order=False, cutoff=None):
    """
    Outlier Tests for RegressionResults instances.

    Parameters
    ----------
    model_results : RegressionResults
        Linear model results
    method : str
        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` :
        - `holm` :
        - `simes-hochberg` :
        - `hommel` :
        - `fdr_bh` : Benjamini/Hochberg
        - `fdr_by` : Benjamini/Yekutieli
        See `statsmodels.stats.multitest.multipletests` for details.
    alpha : float
        familywise error rate
    labels : None or array_like
        If `labels` is not None, then it will be used as index to the
        returned pandas DataFrame. See also Returns below
    order : bool
        Whether or not to order the results by the absolute value of the
        studentized residuals. If labels are provided they will also be sorted.
    cutoff : None or float in [0, 1]
        If cutoff is not None, then the return only includes observations with
        multiple testing corrected p-values strictly below the cutoff. The
        returned array or dataframe can be empty if there are no outlier
        candidates at the specified cutoff.

    Returns
    -------
    table : ndarray or DataFrame
        Returns either an ndarray or a DataFrame if labels is not None.
        Will attempt to get labels from model_results if available. The
        columns are the Studentized residuals, the unadjusted p-value,
        and the corrected p-value according to method.

    Notes
    -----
    The unadjusted p-value is stats.t.sf(abs(resid), df) where
    df = df_resid - 1.
    """
    from scipy import stats
    if labels is None:
        labels = getattr(model_results.model.data, 'row_labels', None)
    infl = getattr(model_results, 'get_influence', None)
    if infl is None:
        results = maybe_unwrap_results(model_results)
        raise AttributeError('model_results object %s does not have a get_influence method.' % results.__class__.__name__)
    resid = infl().resid_studentized_external
    if order:
        idx = np.abs(resid).argsort()[::-1]
        resid = resid[idx]
        if labels is not None:
            labels = np.asarray(labels)[idx]
    df = model_results.df_resid - 1
    unadj_p = stats.t.sf(np.abs(resid), df) * 2
    adj_p = multipletests(unadj_p, alpha=alpha, method=method)
    data = np.c_[resid, unadj_p, adj_p[1]]
    if cutoff is not None:
        mask = data[:, -1] < cutoff
        data = data[mask]
    else:
        mask = slice(None)
    if labels is not None:
        from pandas import DataFrame
        return DataFrame(data, columns=['student_resid', 'unadj_p', method + '(p)'], index=np.asarray(labels)[mask])
    return data