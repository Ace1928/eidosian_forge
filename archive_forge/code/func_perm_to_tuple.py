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
def perm_to_tuple(keys, perm):
    if perm == ():
        d = {k: 0 for k, _ in keys if k is not None}
        return (0,) + tuple(((k, v) for k, v in d.items()))
    d = defaultdict(list)
    y_lags = []
    for v in perm:
        key = keys[v]
        if key[0] is None:
            y_lags.append(key[1])
        else:
            d[key[0]].append(key[1])
    d = dict(d)
    if not y_lags or y_lags == [0]:
        y_lags = 0
    else:
        y_lags = tuple(y_lags)
    for key in keys:
        if key[0] not in d and key[0] is not None:
            d[key[0]] = None
    for key in d:
        if d[key] is not None:
            d[key] = tuple(d[key])
    return (y_lags,) + tuple(((k, v) for k, v in d.items()))