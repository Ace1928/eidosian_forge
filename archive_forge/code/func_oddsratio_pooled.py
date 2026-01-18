import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def oddsratio_pooled(self):
    """
        The pooled odds ratio.

        The value is an estimate of a common odds ratio across all of the
        stratified tables.
        """
    odds_ratio = np.sum(self._ad / self._n) / np.sum(self._bc / self._n)
    return odds_ratio