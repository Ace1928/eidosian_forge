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
@doc_utils.doc_groupby_method(action='compute numerical rank', result='numerical rank', refer_to='rank')
def groupby_rank(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
    return self.groupby_agg(by=by, agg_func='rank', axis=axis, groupby_kwargs=groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)