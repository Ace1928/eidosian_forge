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
@doc_utils.doc_reduce_agg(method='value at the given quantile', refer_to='quantile', params='\n        q : list-like\n        axis : {0, 1}\n        numeric_only : bool\n        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}', extra_params=['**kwargs'])
def quantile_for_list_of_values(self, **kwargs):
    return DataFrameDefault.register(pandas.DataFrame.quantile)(self, **kwargs)