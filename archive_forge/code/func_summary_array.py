import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def summary_array(self, alpha=0.05, use_t=None):
    """Create array with sample statistics and mean estimates

        Parameters
        ----------
        alpha : float in (0, 1)
            Significance level for confidence interval. Nominal coverage is
            ``1 - alpha``.
        use_t : None or bool
            If use_t is None, then the attribute `use_t` determines whether
            normal or t-distribution is used for confidence intervals.
            Specifying use_t overrides the attribute.
            If use_t is false, then confidence intervals are based on the
            normal distribution. If it is true, then the t-distribution is
            used.

        Returns
        -------
        res : ndarray
            Array with columns
            ['eff', "sd_eff", "ci_low", "ci_upp", "w_fe","w_re"].
            Rows include statistics for samples and estimates of overall mean.
        column_names : list of str
            The names for the columns, used when creating summary DataFrame.
        """
    ci_low, ci_upp = self.conf_int_samples(alpha=alpha, use_t=use_t)
    res = np.column_stack([self.eff, self.sd_eff, ci_low, ci_upp, self.weights_rel_fe, self.weights_rel_re])
    ci = self.conf_int(alpha=alpha, use_t=use_t)
    res_fe = [[self.mean_effect_fe, self.sd_eff_w_fe, ci[0][0], ci[0][1], 1, np.nan]]
    res_re = [[self.mean_effect_re, self.sd_eff_w_re, ci[1][0], ci[1][1], np.nan, 1]]
    res_fe_wls = [[self.mean_effect_fe, self.sd_eff_w_fe_hksj, ci[2][0], ci[2][1], 1, np.nan]]
    res_re_wls = [[self.mean_effect_re, self.sd_eff_w_re_hksj, ci[3][0], ci[3][1], np.nan, 1]]
    res = np.concatenate([res, res_fe, res_re, res_fe_wls, res_re_wls], axis=0)
    column_names = ['eff', 'sd_eff', 'ci_low', 'ci_upp', 'w_fe', 'w_re']
    return (res, column_names)