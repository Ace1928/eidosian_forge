from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
from collections.abc import Hashable, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)
def check_exog(arr, name, orig, exact):
    if isinstance(orig, pd.DataFrame):
        if not isinstance(arr, pd.DataFrame):
            raise TypeError(f'{name} must be a DataFrame when the original exog was a DataFrame')
        if sorted(arr.columns) != sorted(self.data.orig_exog.columns):
            raise ValueError(f'{name} must have the same columns as the original exog')
    else:
        arr = array_like(arr, name, ndim=2, optional=False)
    if arr.ndim != 2 or arr.shape[1] != orig.shape[1]:
        raise ValueError(f'{name} must have the same number of columns as the original data, {orig.shape[1]}')
    if exact and arr.shape[0] != orig.shape[0]:
        raise ValueError(f'{name} must have the same number of rows as the original data ({n}).')
    return arr