import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def riskratio(self):
    """
        Returns the risk ratio for a 2x2 table.

        The risk ratio is calculated with respect to the rows.
        """
    p = self.table[:, 0] / self.table.sum(1)
    return p[0] / p[1]