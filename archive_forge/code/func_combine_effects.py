import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def combine_effects(effect, variance, method_re='iterated', row_names=None, use_t=False, alpha=0.05, **kwds):
    """combining effect sizes for effect sizes using meta-analysis

    This currently does not use np.asarray, all computations are possible in
    pandas.

    Parameters
    ----------
    effect : array
        mean of effect size measure for all samples
    variance : array
        variance of mean or effect size measure for all samples
    method_re : {"iterated", "chi2"}
        method that is use to compute the between random effects variance
        "iterated" or "pm" uses Paule and Mandel method to iteratively
        estimate the random effects variance. Options for the iteration can
        be provided in the ``kwds``
        "chi2" or "dl" uses DerSimonian and Laird one-step estimator.
    row_names : list of strings (optional)
        names for samples or studies, will be included in results summary and
        table.
    alpha : float in (0, 1)
        significance level, default is 0.05, for the confidence intervals

    Returns
    -------
    results : CombineResults
        Contains estimation results and intermediate statistics, and includes
        a method to return a summary table.
        Statistics from intermediate calculations might be removed at a later
        time.

    Notes
    -----
    Status: Basic functionality is verified, mainly compared to R metafor
    package. However, API might still change.

    This computes both fixed effects and random effects estimates. The
    random effects results depend on the method to estimate the RE variance.

    Scale estimate
    In fixed effects models and in random effects models without fully
    iterated random effects variance, the model will in general not account
    for all residual variance. Traditional meta-analysis uses a fixed
    scale equal to 1, that might not produce test statistics and
    confidence intervals with the correct size. Estimating the scale to account
    for residual variance often improves the small sample properties of
    inference and confidence intervals.
    This adjustment to the standard errors is often referred to as HKSJ
    method based attributed to Hartung and Knapp and Sidik and Jonkman.
    However, this is equivalent to estimating the scale in WLS.
    The results instance includes both, fixed scale and estimated scale
    versions of standard errors and confidence intervals.

    References
    ----------
    Borenstein, Michael. 2009. Introduction to Meta-Analysis.
        Chichester: Wiley.

    Chen, Ding-Geng, and Karl E. Peace. 2013. Applied Meta-Analysis with R.
        Chapman & Hall/CRC Biostatistics Series.
        Boca Raton: CRC Press/Taylor & Francis Group.

    """
    k = len(effect)
    if row_names is None:
        row_names = list(range(k))
    crit = stats.norm.isf(alpha / 2)
    eff = effect
    var_eff = variance
    sd_eff = np.sqrt(var_eff)
    weights_fe = 1 / var_eff
    w_total_fe = weights_fe.sum(0)
    weights_rel_fe = weights_fe / w_total_fe
    eff_w_fe = weights_rel_fe * eff
    mean_effect_fe = eff_w_fe.sum()
    var_eff_w_fe = 1 / w_total_fe
    sd_eff_w_fe = np.sqrt(var_eff_w_fe)
    q = (weights_fe * eff ** 2).sum(0)
    q -= (weights_fe * eff).sum() ** 2 / w_total_fe
    df = k - 1
    if method_re.lower() in ['iterated', 'pm']:
        tau2, _ = _fit_tau_iterative(eff, var_eff, **kwds)
    elif method_re.lower() in ['chi2', 'dl']:
        c = w_total_fe - (weights_fe ** 2).sum() / w_total_fe
        tau2 = (q - df) / c
    else:
        raise ValueError('method_re should be "iterated" or "chi2"')
    weights_re = 1 / (var_eff + tau2)
    w_total_re = weights_re.sum(0)
    weights_rel_re = weights_re / weights_re.sum(0)
    eff_w_re = weights_rel_re * eff
    mean_effect_re = eff_w_re.sum()
    var_eff_w_re = 1 / w_total_re
    sd_eff_w_re = np.sqrt(var_eff_w_re)
    scale_hksj_re = (weights_re * (eff - mean_effect_re) ** 2).sum() / df
    scale_hksj_fe = (weights_fe * (eff - mean_effect_fe) ** 2).sum() / df
    var_hksj_re = (weights_rel_re * (eff - mean_effect_re) ** 2).sum() / df
    var_hksj_fe = (weights_rel_fe * (eff - mean_effect_fe) ** 2).sum() / df
    res = CombineResults(**locals())
    return res