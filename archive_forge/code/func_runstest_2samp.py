import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def runstest_2samp(x, y=None, groups=None, correction=True):
    """Wald-Wolfowitz runstest for two samples

    This tests whether two samples come from the same distribution.

    Parameters
    ----------
    x : array_like
        data, numeric, contains either one group, if y is also given, or
        both groups, if additionally a group indicator is provided
    y : array_like (optional)
        data, numeric
    groups : array_like
        group labels or indicator the data for both groups is given in a
        single 1-dimensional array, x. If group labels are not [0,1], then
    correction : bool
        Following the SAS manual, for samplesize below 50, the test
        statistic is corrected by 0.5. This can be turned off with
        correction=False, and was included to match R, tseries, which
        does not use any correction.

    Returns
    -------
    z_stat : float
        test statistic, asymptotically normally distributed
    p-value : float
        p-value, reject the null hypothesis if it is below an type 1 error
        level, alpha .


    Notes
    -----
    Wald-Wolfowitz runs test.

    If there are ties, then then the test statistic and p-value that is
    reported, is based on the higher p-value between sorting all tied
    observations of the same group


    This test is intended for continuous distributions
    SAS has treatment for ties, but not clear, and sounds more complicated
    (minimum and maximum possible runs prevent use of argsort)
    (maybe it's not so difficult, idea: add small positive noise to first
    one, run test, then to the other, run test, take max(?) p-value - DONE
    This gives not the minimum and maximum of the number of runs, but should
    be close. Not true, this is close to minimum but far away from maximum.
    maximum number of runs would use alternating groups in the ties.)
    Maybe adding random noise would be the better approach.

    SAS has exact distribution for sample size <=30, does not look standard
    but should be easy to add.

    currently two-sided test only

    This has not been verified against a reference implementation. In a short
    Monte Carlo simulation where both samples are normally distribute, the test
    seems to be correctly sized for larger number of observations (30 or
    larger), but conservative (i.e. reject less often than nominal) with a
    sample size of 10 in each group.

    See Also
    --------
    runs_test_1samp
    Runs
    RunsProb

    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
        groups = np.concatenate((np.zeros(len(x)), np.ones(len(y))))
        x = np.concatenate((x, y))
        gruni = np.arange(2)
    elif groups is not None:
        gruni = np.unique(groups)
        if gruni.size != 2:
            raise ValueError('not exactly two groups specified')
    else:
        raise ValueError('either y or groups is necessary')
    xargsort = np.argsort(x)
    x_sorted = x[xargsort]
    x_diff = np.diff(x_sorted)
    if x_diff.min() == 0:
        print('ties detected')
        x_mindiff = x_diff[x_diff > 0].min()
        eps = x_mindiff / 2.0
        xx = x.copy()
        xx[groups == gruni[0]] += eps
        xargsort = np.argsort(xx)
        xindicator = groups[xargsort]
        z0, p0 = Runs(xindicator).runs_test(correction=correction)
        xx[groups == gruni[0]] -= eps
        xx[groups == gruni[1]] += eps
        xargsort = np.argsort(xx)
        xindicator = groups[xargsort]
        z1, p1 = Runs(xindicator).runs_test(correction=correction)
        idx = np.argmax([p0, p1])
        return ([z0, z1][idx], [p0, p1][idx])
    else:
        xindicator = groups[xargsort]
        return Runs(xindicator).runs_test(correction=correction)