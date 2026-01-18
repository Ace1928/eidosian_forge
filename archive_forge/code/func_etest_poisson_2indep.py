import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def etest_poisson_2indep(count1, exposure1, count2, exposure2, ratio_null=None, value=None, method='score', compare='ratio', alternative='two-sided', ygrid=None, y_grid=None):
    """
    E-test for ratio of two sample Poisson rates.

    Rates are defined as expected count divided by exposure. The Null and
    alternative hypothesis for the rates, rate1 and rate2, of two independent
    Poisson samples are:

    for compare = 'diff'

    - H0: rate1 - rate2 - value = 0
    - H1: rate1 - rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 - rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 - rate2 - value < 0   if alternative = 'smaller'

    for compare = 'ratio'

    - H0: rate1 / rate2 - value = 0
    - H1: rate1 / rate2 - value != 0  if alternative = 'two-sided'
    - H1: rate1 / rate2 - value > 0   if alternative = 'larger'
    - H1: rate1 / rate2 - value < 0   if alternative = 'smaller'

    Parameters
    ----------
    count1 : int
        Number of events in first sample
    exposure1 : float
        Total exposure (time * subjects) in first sample
    count2 : int
        Number of events in first sample
    exposure2 : float
        Total exposure (time * subjects) in first sample
    ratio_null: float
        Ratio of the two Poisson rates under the Null hypothesis. Default is 1.
        Deprecated, use ``value`` instead.

        .. deprecated:: 0.14.0

            Use ``value`` instead.

    value : float
        Value of the ratio or diff of 2 independent rates under the null
        hypothesis. Default is equal rates, i.e. 1 for ratio and 0 for diff.

        .. versionadded:: 0.14.0

            Replacement for ``ratio_null``.

    method : {"score", "wald"}
        Method for the test statistic that defines the rejection region.
    alternative : string
        The alternative hypothesis, H1, has to be one of the following

        - 'two-sided': H1: ratio of rates is not equal to ratio_null (default)
        - 'larger' :   H1: ratio of rates is larger than ratio_null
        - 'smaller' :  H1: ratio of rates is smaller than ratio_null

    y_grid : None or 1-D ndarray
        Grid values for counts of the Poisson distribution used for computing
        the pvalue. By default truncation is based on an upper tail Poisson
        quantiles.

    ygrid : None or 1-D ndarray
        Same as y_grid. Deprecated. If both y_grid and ygrid are provided,
        ygrid will be ignored.

        .. deprecated:: 0.14.0

            Use ``y_grid`` instead.

    Returns
    -------
    stat_sample : float
        test statistic for the sample
    pvalue : float

    References
    ----------
    Gu, Ng, Tang, Schucany 2008: Testing the Ratio of Two Poisson Rates,
    Biometrical Journal 50 (2008) 2, 2008
    Ng, H. K. T., K. Gu, and M. L. Tang. 2007. “A Comparative Study of Tests
    for the Difference of Two Poisson Means.” Computational Statistics & Data
    Analysis 51 (6): 3085–99. https://doi.org/10.1016/j.csda.2006.02.004.

    """
    y1, n1, y2, n2 = map(np.asarray, [count1, exposure1, count2, exposure2])
    d = n2 / n1
    eps = 1e-20
    if compare == 'ratio':
        if ratio_null is None and value is None:
            value = 1
        elif ratio_null is not None:
            warnings.warn("'ratio_null' is deprecated, use 'value' keyword", FutureWarning)
            value = ratio_null
        r = value
        r_d = r / d
        rate2_cmle = (y1 + y2) / n2 / (1 + r_d)
        rate1_cmle = rate2_cmle * r
        if method in ['score']:

            def stat_func(x1, x2):
                return (x1 - x2 * r_d) / np.sqrt((x1 + x2) * r_d + eps)
        elif method in ['wald']:

            def stat_func(x1, x2):
                return (x1 - x2 * r_d) / np.sqrt(x1 + x2 * r_d ** 2 + eps)
        else:
            raise ValueError('method not recognized')
    elif compare == 'diff':
        if value is None:
            value = 0
        tmp = _score_diff(y1, n1, y2, n2, value=value, return_cmle=True)
        _, rate1_cmle, rate2_cmle = tmp
        if method in ['score']:

            def stat_func(x1, x2):
                return _score_diff(x1, n1, x2, n2, value=value)
        elif method in ['wald']:

            def stat_func(x1, x2):
                rate1, rate2 = (x1 / n1, x2 / n2)
                stat = rate1 - rate2 - value
                stat /= np.sqrt(rate1 / n1 + rate2 / n2 + eps)
                return stat
        else:
            raise ValueError('method not recognized')
    rate1 = rate1_cmle
    rate2 = rate2_cmle
    mean1 = n1 * rate1
    mean2 = n2 * rate2
    stat_sample = stat_func(y1, y2)
    if ygrid is not None:
        warnings.warn('ygrid is deprecated, use y_grid', FutureWarning)
    y_grid = y_grid if y_grid is not None else ygrid
    if y_grid is None:
        threshold = stats.poisson.isf(1e-13, max(mean1, mean2))
        threshold = max(threshold, 100)
        y_grid = np.arange(threshold + 1)
    else:
        y_grid = np.asarray(y_grid)
        if y_grid.ndim != 1:
            raise ValueError('y_grid needs to be None or 1-dimensional array')
    pdf1 = stats.poisson.pmf(y_grid, mean1)
    pdf2 = stats.poisson.pmf(y_grid, mean2)
    stat_space = stat_func(y_grid[:, None], y_grid[None, :])
    eps = 1e-15
    if alternative in ['two-sided', '2-sided', '2s']:
        mask = np.abs(stat_space) >= np.abs(stat_sample) - eps
    elif alternative in ['larger', 'l']:
        mask = stat_space >= stat_sample - eps
    elif alternative in ['smaller', 's']:
        mask = stat_space <= stat_sample + eps
    else:
        raise ValueError('invalid alternative')
    pvalue = (pdf1[:, None] * pdf2[None, :])[mask].sum()
    return (stat_sample, pvalue)