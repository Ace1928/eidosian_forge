import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def local_log_oddsratios(self):
    """
        Returns local log odds ratios.

        The local log odds ratios are the log odds ratios
        calculated for contiguous 2x2 sub-tables.
        """
    ta = self.table.copy()
    a = ta[0:-1, 0:-1]
    b = ta[0:-1, 1:]
    c = ta[1:, 0:-1]
    d = ta[1:, 1:]
    tab = np.log(a) + np.log(d) - np.log(b) - np.log(c)
    rslt = np.empty(self.table.shape, np.float64)
    rslt *= np.nan
    rslt[0:-1, 0:-1] = tab
    if isinstance(self.table_orig, pd.DataFrame):
        rslt = pd.DataFrame(rslt, index=self.table_orig.index, columns=self.table_orig.columns)
    return rslt