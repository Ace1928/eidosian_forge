import abc
import warnings
from typing import Hashable, List, Optional
import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import DtypeBackend, IndexLabel, Suffixes
from pandas.core.dtypes.common import is_number, is_scalar
from modin.config import StorageFormat
from modin.core.dataframe.algebra.default2pandas import (
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
from . import doc_utils
@doc_utils.add_deprecation_warning(replacement_method='rolling_aggregate')
@doc_utils.doc_window_method(window_cls_name='Rolling', result='the result of passed function', action='apply specified function', refer_to='apply', params='\n        func : callable(pandas.Series) -> scalar\n        raw : bool, default: False\n        engine : None, default: None\n            This parameters serves the compatibility purpose. Always has to be None.\n        engine_kwargs : None, default: None\n            This parameters serves the compatibility purpose. Always has to be None.\n        args : tuple, optional\n        kwargs : dict, optional', build_rules='udf_aggregation')
def rolling_apply(self, fold_axis, rolling_kwargs, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None):
    return RollingDefault.register(pandas.core.window.rolling.Rolling.apply)(self, rolling_kwargs, func, raw, engine, engine_kwargs, args, kwargs)