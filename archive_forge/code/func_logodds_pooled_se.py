import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def logodds_pooled_se(self):
    """
        Estimated standard error of the pooled log odds ratio

        References
        ----------
        J. Robins, N. Breslow, S. Greenland. "Estimators of the
        Mantel-Haenszel Variance Consistent in Both Sparse Data and
        Large-Strata Limiting Models." Biometrics 42, no. 2 (1986): 311-23.
        """
    adns = np.sum(self._ad / self._n)
    bcns = np.sum(self._bc / self._n)
    lor_va = np.sum(self._apd * self._ad / self._n ** 2) / adns ** 2
    mid = self._apd * self._bc / self._n ** 2
    mid += (1 - self._apd / self._n) * self._ad / self._n
    mid = np.sum(mid)
    mid /= adns * bcns
    lor_va += mid
    lor_va += np.sum((1 - self._apd / self._n) * self._bc / self._n) / bcns ** 2
    lor_va /= 2
    lor_se = np.sqrt(lor_va)
    return lor_se