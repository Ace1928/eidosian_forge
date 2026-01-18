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
def model_orders(self):
    """
        The orders of each of the polynomials in the model.
        """
    return {'trend': self._k_trend, 'exog': self._k_exog, 'ar': self.k_ar, 'ma': self.k_ma, 'seasonal_ar': self.k_seasonal_ar, 'seasonal_ma': self.k_seasonal_ma, 'reduced_ar': self.k_ar + self.k_seasonal_ar, 'reduced_ma': self.k_ma + self.k_seasonal_ma, 'exog_variance': self._k_exog if self.state_regression and self.time_varying_regression else 0, 'measurement_variance': int(self.measurement_error), 'variance': int(self.state_error and (not self.concentrate_scale))}