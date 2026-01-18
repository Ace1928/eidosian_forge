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
def param_terms(self):
    """
        List of parameters actually included in the model, in sorted order.

        TODO Make this an dict with slice or indices as the values.
        """
    model_orders = self.model_orders
    params = [order for order in self.params_complete if model_orders[order] > 0]
    if 'exog' in params and (not self.mle_regression):
        params.remove('exog')
    return params