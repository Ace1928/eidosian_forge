import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def independence_probabilities(self):
    """
        Returns fitted joint probabilities under independence.

        The returned table is outer(row, column), where row and
        column are the estimated marginal distributions
        of the rows and columns.
        """
    row, col = self.marginal_probabilities
    itab = np.outer(row, col)
    if isinstance(self.table_orig, pd.DataFrame):
        itab = pd.DataFrame(itab, self.table_orig.index, self.table_orig.columns)
    return itab