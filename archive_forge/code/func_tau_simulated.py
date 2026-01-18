from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from statsmodels.graphics import utils
def tau_simulated(self, nobs=1024, random_state=None):
    """Kendall's tau based on simulated samples.

        Returns
        -------
        tau : float
            Kendall's tau.

        """
    x = self.rvs(nobs, random_state=random_state)
    return stats.kendalltau(x[:, 0], x[:, 1])[0]