import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def riskratio_pooled(self):
    """
        Estimate of the pooled risk ratio.
        """
    acd = self.table[0, 0, :] * self._cpd
    cab = self.table[1, 0, :] * self._apb
    rr = np.sum(acd / self._n) / np.sum(cab / self._n)
    return rr