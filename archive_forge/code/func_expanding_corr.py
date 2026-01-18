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
@doc_utils.doc_window_method(window_cls_name='Expanding', result='correlation', refer_to='corr', win_type='expanding window', params='\n        squeeze_self : bool\n        squeeze_other : bool\n        other : pandas.Series or pandas.DataFrame, default: None\n        pairwise : bool | None, default: None\n        ddof : int, default: 1\n        numeric_only : bool, default: False\n        **kwargs : dict')
def expanding_corr(self, fold_axis, expanding_args, squeeze_self, squeeze_other, other=None, pairwise=None, ddof=1, numeric_only=False, **kwargs):
    other_for_default = other if other is None else other.to_pandas().squeeze(axis=1) if squeeze_other else other.to_pandas()
    return ExpandingDefault.register(pandas.core.window.expanding.Expanding.corr, squeeze_self=squeeze_self)(self, expanding_args, other=other_for_default, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only, **kwargs)