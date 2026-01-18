import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import (
from scipy.stats.distributions import rv_frozen
from statsmodels.base.data import PandasData
from statsmodels.base.model import Results
from statsmodels.base.wrapper import (
@property
def sse(self):
    """
        The sum of squared errors between the data and the fittted value.
        """
    return self._sse