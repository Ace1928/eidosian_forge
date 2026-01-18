import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def tconfint_diff(self, alpha=0.05, alternative='two-sided', usevar='pooled'):
    """confidence interval for the difference in means

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :

            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        lower, upper : floats
            lower and upper limits of the confidence interval

        Notes
        -----
        The result is independent of the user specified ddof.

        """
    d1 = self.d1
    d2 = self.d2
    diff = d1.mean - d2.mean
    if usevar == 'pooled':
        std_diff = self.std_meandiff_pooledvar
        dof = d1.nobs - 1 + d2.nobs - 1
    elif usevar == 'unequal':
        std_diff = self.std_meandiff_separatevar
        dof = self.dof_satt()
    else:
        raise ValueError('usevar can only be "pooled" or "unequal"')
    res = _tconfint_generic(diff, std_diff, dof, alpha=alpha, alternative=alternative)
    return res