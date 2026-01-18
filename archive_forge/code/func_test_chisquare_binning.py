import numpy as np
from scipy import stats
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.effect_size import _noncentrality_chisquare
def test_chisquare_binning(counts, expected, sort_var=None, bins=10, df=None, ordered=False, sort_method='quicksort', alpha_nc=0.05):
    """chisquare gof test with binning of data, Hosmer-Lemeshow type

    ``observed`` and ``expected`` are observation specific and should have
    observations in rows and choices in columns

    Parameters
    ----------
    counts : array_like
        Observed frequency, i.e. counts for all choices
    expected : array_like
        Expected counts or probability. If expected are counts, then they
        need to sum to the same total count as the sum of observed.
        If those sums are unequal and all expected values are smaller or equal
        to 1, then they are interpreted as probabilities and will be rescaled
        to match counts.
    sort_var : array_like
        1-dimensional array for binning. Groups will be formed according to
        quantiles of the sorted array ``sort_var``, so that group sizes have
        equal or approximately equal sizes.

    Returns
    -------
    Holdertuple instance
        This instance contains the results of the chisquare test and some
        information about the data

        - statistic : chisquare statistic of the goodness-of-fit test
        - pvalue : pvalue of the chisquare test
        = df : degrees of freedom of the test

    Notes
    -----
    Degrees of freedom for Hosmer-Lemeshow tests are given by

    g groups, c choices

    - binary: `df = (g - 2)` for insample,
         Stata uses `df = g` for outsample
    - multinomial: `df = (g−2) *(c−1)`, reduces to (g-2) for binary c=2,
         (Fagerland, Hosmer, Bofin SIM 2008)
    - ordinal: `df = (g - 2) * (c - 1) + (c - 2)`, reduces to (g-2) for c=2,
         (Hosmer, ... ?)

    Note: If there are ties in the ``sort_var`` array, then the split of
    observations into groups will depend on the sort algorithm.
    """
    observed = np.asarray(counts)
    expected = np.asarray(expected)
    n_observed = counts.sum()
    n_expected = expected.sum()
    if not np.allclose(n_observed, n_expected, atol=1e-13):
        if np.max(expected) < 1 + 1e-13:
            import warnings
            warnings.warn('sum of expected and of observed differ, rescaling ``expected``')
            expected = expected / n_expected * n_observed
        else:
            raise ValueError('total counts of expected and observed differ')
    if sort_var is not None:
        argsort = np.argsort(sort_var, kind=sort_method)
    else:
        argsort = np.arange(observed.shape[0])
    indices = np.array_split(argsort, bins, axis=0)
    freqs = np.array([observed[idx].sum(0) for idx in indices])
    probs = np.array([expected[idx].sum(0) for idx in indices])
    resid_pearson = (freqs - probs) / np.sqrt(probs)
    chi2_stat_groups = ((freqs - probs) ** 2 / probs).sum(1)
    chi2_stat = chi2_stat_groups.sum()
    if df is None:
        g, c = freqs.shape
        if ordered is True:
            df = (g - 2) * (c - 1) + (c - 2)
        else:
            df = (g - 2) * (c - 1)
    pvalue = stats.chi2.sf(chi2_stat, df)
    noncentrality = _noncentrality_chisquare(chi2_stat, df, alpha=alpha_nc)
    res = HolderTuple(statistic=chi2_stat, pvalue=pvalue, df=df, freqs=freqs, probs=probs, noncentrality=noncentrality, resid_pearson=resid_pearson, chi2_stat_groups=chi2_stat_groups, indices=indices)
    return res