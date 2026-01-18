from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS, GLS, RegressionResults
from statsmodels.regression.feasible_gls import atleast_2dcols
def share_data(self):
    """a measure for the fraction of the data in the estimation result

        The share of the prior information is `1 - share_data`.

        Returns
        -------
        share : float between 0 and 1
            share of data defined as the ration between effective degrees of
            freedom of the model and the number (TODO should be rank) of the
            explanatory variables.
        """
    return (self.df_model + 1) / self.model.rank