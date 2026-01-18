from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_invalid_kwargs():
    endog = [0, 0, 1.0]
    sarimax.SARIMAX(endog)
    with pytest.warns(FutureWarning):
        sarimax.SARIMAX(endog, invalid_kwarg=True)