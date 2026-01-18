import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def effectsize_normal(self, prob=None):
    """
        Cohen's d, standardized mean difference under normality assumption.

        This computes the standardized mean difference, Cohen's d, effect size
        that is equivalent to the rank based probability ``p`` of being
        stochastically larger if we assume that the data is normally
        distributed, given by

            :math: `d = F^{-1}(p) * \\sqrt{2}`

        where :math:`F^{-1}` is the inverse of the cdf of the normal
        distribution.

        Parameters
        ----------
        prob : float in (0, 1)
            Probability to be converted to Cohen's d effect size.
            If prob is None, then the ``prob1`` attribute is used.

        Returns
        -------
        equivalent Cohen's d effect size under normality assumption.

        """
    if prob is None:
        prob = self.prob1
    return stats.norm.ppf(prob) * np.sqrt(2)