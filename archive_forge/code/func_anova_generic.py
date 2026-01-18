import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def anova_generic(means, variances, nobs, use_var='unequal', welch_correction=True, info=None):
    """
    Oneway Anova based on summary statistics

    Parameters
    ----------
    means : array_like
        Mean of samples to be compared
    variances : float or array_like
        Residual (within) variance of each sample or pooled.
        If ``variances`` is scalar, then it is interpreted as pooled variance
        that is the same for all samples, ``use_var`` will be ignored.
        Otherwise, the variances are used depending on the ``use_var`` keyword.
    nobs : int or array_like
        Number of observations for the samples.
        If nobs is scalar, then it is assumed that all samples have the same
        number ``nobs`` of observation, i.e. a balanced sample case.
        Otherwise, statistics will be weighted corresponding to nobs.
        Only relative sizes are relevant, any proportional change to nobs does
        not change the effect size.
    use_var : {"unequal", "equal", "bf"}
        If ``use_var`` is "unequal", then the variances can differ across
        samples and the effect size for Welch anova will be computed.
    welch_correction : bool
        If this is false, then the Welch correction to the test statistic is
        not included. This allows the computation of an effect size measure
        that corresponds more closely to Cohen's f.
    info : not used yet

    Returns
    -------
    res : results instance
        This includes `statistic` and `pvalue`.

    """
    options = {'use_var': use_var, 'welch_correction': welch_correction}
    if means.ndim != 1:
        raise ValueError('data (means, ...) has to be one-dimensional')
    nobs_t = nobs.sum()
    n_groups = len(means)
    if use_var == 'unequal':
        weights = nobs / variances
    else:
        weights = nobs
    w_total = weights.sum()
    w_rel = weights / w_total
    meanw_t = w_rel @ means
    statistic = np.dot(weights, (means - meanw_t) ** 2) / (n_groups - 1.0)
    df_num = n_groups - 1.0
    if use_var == 'unequal':
        tmp = ((1 - w_rel) ** 2 / (nobs - 1)).sum() / (n_groups ** 2 - 1)
        if welch_correction:
            statistic /= 1 + 2 * (n_groups - 2) * tmp
        df_denom = 1.0 / (3.0 * tmp)
    elif use_var == 'equal':
        tmp = ((nobs - 1) * variances).sum() / (nobs_t - n_groups)
        statistic /= tmp
        df_denom = nobs_t - n_groups
    elif use_var == 'bf':
        tmp = ((1.0 - nobs / nobs_t) * variances).sum()
        statistic = 1.0 * (nobs * (means - meanw_t) ** 2).sum()
        statistic /= tmp
        df_num2 = n_groups - 1
        df_denom = tmp ** 2 / ((1.0 - nobs / nobs_t) ** 2 * variances ** 2 / (nobs - 1)).sum()
        df_num = tmp ** 2 / ((variances ** 2).sum() + (nobs / nobs_t * variances).sum() ** 2 - 2 * (nobs / nobs_t * variances ** 2).sum())
        pval2 = stats.f.sf(statistic, df_num2, df_denom)
        options['df2'] = (df_num2, df_denom)
        options['df_num2'] = df_num2
        options['pvalue2'] = pval2
    else:
        raise ValueError('use_var is to be one of "unequal", "equal" or "bf"')
    pval = stats.f.sf(statistic, df_num, df_denom)
    res = HolderTuple(statistic=statistic, pvalue=pval, df=(df_num, df_denom), df_num=df_num, df_denom=df_denom, nobs_t=nobs_t, n_groups=n_groups, means=means, nobs=nobs, vars_=variances, **options)
    return res