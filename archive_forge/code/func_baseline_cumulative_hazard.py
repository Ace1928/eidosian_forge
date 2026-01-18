import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
@cache_readonly
def baseline_cumulative_hazard(self):
    """
        A list (corresponding to the strata) containing the baseline
        cumulative hazard function evaluated at the event points.
        """
    return self.model.baseline_cumulative_hazard(self.params)