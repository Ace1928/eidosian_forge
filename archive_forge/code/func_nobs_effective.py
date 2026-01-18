from collections import OrderedDict
import contextlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.base.data import PandasData
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
from statsmodels.tools.sm_exceptions import PrecisionWarning
from statsmodels.tools.numdiff import (
from statsmodels.tools.tools import pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.statespace.tools import _safe_cond
@cache_readonly
def nobs_effective(self):
    raise NotImplementedError