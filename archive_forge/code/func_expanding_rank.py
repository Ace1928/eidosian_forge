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
@doc_utils.doc_window_method(window_cls_name='Expanding', result='rank', refer_to='rank', win_type='expanding window', params="\n        method : {'average', 'min', 'max'}, default: 'average'\n        ascending : bool, default: True\n        pct : bool, default: False\n        numeric_only : bool, default: False\n        *args : iterable\n        **kwargs : dict")
def expanding_rank(self, fold_axis, expanding_args, method='average', ascending=True, pct=False, numeric_only=False, *args, **kwargs):
    return ExpandingDefault.register(pandas.core.window.expanding.Expanding.rank)(self, expanding_args, *args, method=method, ascending=ascending, pct=pct, numeric_only=numeric_only, **kwargs)