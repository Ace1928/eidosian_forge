import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def cumulative_log_oddsratios(self):
    """
        Returns cumulative log odds ratios.

        The cumulative log odds ratios for a contingency table
        with ordered rows and columns are calculated by collapsing
        all cells to the left/right and above/below a given point,
        to obtain a 2x2 table from which a log odds ratio can be
        calculated.
        """
    ta = self.table.cumsum(0).cumsum(1)
    a = ta[0:-1, 0:-1]
    b = ta[0:-1, -1:] - a
    c = ta[-1:, 0:-1] - a
    d = ta[-1, -1] - (a + b + c)
    tab = np.log(a) + np.log(d) - np.log(b) - np.log(c)
    rslt = np.empty(self.table.shape, np.float64)
    rslt *= np.nan
    rslt[0:-1, 0:-1] = tab
    if isinstance(self.table_orig, pd.DataFrame):
        rslt = pd.DataFrame(rslt, index=self.table_orig.index, columns=self.table_orig.columns)
    return rslt