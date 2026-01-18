from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def short_model_name(error, trend, seasonal, damped=False):
    short_name = {'add': 'A', 'mul': 'M', None: 'N', True: 'd', False: ''}
    return short_name[error] + short_name[trend] + short_name[damped] + short_name[seasonal]