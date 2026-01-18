from warnings import warn
import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.tsatools import lagmat
from .initialization import Initialization
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
@property
def initial_selection(self):
    """Initial selection matrix"""
    if not (self.state_regression and self.time_varying_regression):
        if self.k_posdef > 0:
            selection = np.r_[[0] * self._k_states_diff, [1] * (self._k_order > 0), [0] * (self._k_order - 1), [0] * ((1 - self.mle_regression) * self._k_exog)][:, None]
            if len(selection) == 0:
                selection = np.zeros((self.k_states, self.k_posdef))
        else:
            selection = np.zeros((self.k_states, 0))
    else:
        selection = np.zeros((self.k_states, self.k_posdef))
        if self._k_order > 0:
            selection[0, 0] = 1
        for i in range(self._k_exog, 0, -1):
            selection[-i, -i] = 1
    return selection