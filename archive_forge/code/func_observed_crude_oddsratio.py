from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def observed_crude_oddsratio(self):
    """
        To obtain the crude (global) odds ratio, first pool all binary
        indicators corresponding to a given pair of cut points (c,c'),
        then calculate the odds ratio for this 2x2 table.  The crude
        odds ratio is the inverse variance weighted average of these
        odds ratios.  Since the covariate effects are ignored, this OR
        will generally be greater than the stratified OR.
        """
    cpp = self.cpp
    endog = self.model.endog_li
    tables = {}
    for ii in cpp[0].keys():
        tables[ii] = np.zeros((2, 2), dtype=np.float64)
    for i in range(len(endog)):
        yvec = endog[i]
        endog_11 = np.outer(yvec, yvec)
        endog_10 = np.outer(yvec, 1.0 - yvec)
        endog_01 = np.outer(1.0 - yvec, yvec)
        endog_00 = np.outer(1.0 - yvec, 1.0 - yvec)
        cpp1 = cpp[i]
        for ky in cpp1.keys():
            ix = cpp1[ky]
            tables[ky][1, 1] += endog_11[ix[:, 0], ix[:, 1]].sum()
            tables[ky][1, 0] += endog_10[ix[:, 0], ix[:, 1]].sum()
            tables[ky][0, 1] += endog_01[ix[:, 0], ix[:, 1]].sum()
            tables[ky][0, 0] += endog_00[ix[:, 0], ix[:, 1]].sum()
    return self.pooled_odds_ratio(list(tables.values()))