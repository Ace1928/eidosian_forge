import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
@cache_readonly
def baseline_cumulative_hazard_function(self):
    """
        A list (corresponding to the strata) containing function
        objects that calculate the cumulative hazard function.
        """
    return self.model.baseline_cumulative_hazard_function(self.params)