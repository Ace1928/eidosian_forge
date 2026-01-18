import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def effectsize_smd(mean1, sd1, nobs1, mean2, sd2, nobs2):
    """effect sizes for mean difference for use in meta-analysis

    mean1, sd1, nobs1 are for treatment
    mean2, sd2, nobs2 are for control

    Effect sizes are computed for the mean difference ``mean1 - mean2``
    standardized by an estimate of the within variance.

    This does not have option yet.
    It uses standardized mean difference with bias correction as effect size.

    This currently does not use np.asarray, all computations are possible in
    pandas.

    Parameters
    ----------
    mean1 : array
        mean of second sample, treatment groups
    sd1 : array
        standard deviation of residuals in treatment groups, within
    nobs1 : array
        number of observations in treatment groups
    mean2, sd2, nobs2 : arrays
        mean, standard deviation and number of observations of control groups

    Returns
    -------
    smd_bc : array
        bias corrected estimate of standardized mean difference
    var_smdbc : array
        estimate of variance of smd_bc

    Notes
    -----
    Status: API will still change. This is currently intended for support of
    meta-analysis.

    References
    ----------
    Borenstein, Michael. 2009. Introduction to Meta-Analysis.
        Chichester: Wiley.

    Chen, Ding-Geng, and Karl E. Peace. 2013. Applied Meta-Analysis with R.
        Chapman & Hall/CRC Biostatistics Series.
        Boca Raton: CRC Press/Taylor & Francis Group.

    """
    var_diff = (sd1 ** 2 * (nobs1 - 1) + sd2 ** 2 * (nobs2 - 1)) / (nobs1 + nobs2 - 2)
    sd_diff = np.sqrt(var_diff)
    nobs = nobs1 + nobs2
    bias_correction = 1 - 3 / (4 * nobs - 9)
    smd = (mean1 - mean2) / sd_diff
    smd_bc = bias_correction * smd
    var_smdbc = nobs / nobs1 / nobs2 + smd_bc ** 2 / 2 / (nobs - 3.94)
    return (smd_bc, var_smdbc)