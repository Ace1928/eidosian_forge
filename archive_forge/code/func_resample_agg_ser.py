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
@doc_utils.add_deprecation_warning(replacement_method='resample_agg_df')
@doc_utils.doc_resample_agg(action='apply passed aggregation function in a one-column query compiler', params='func : str, dict, callable(pandas.Series) -> scalar, or list of such', output='function names', refer_to='agg')
def resample_agg_ser(self, resample_kwargs, func, *args, **kwargs):
    return ResampleDefault.register(pandas.core.resample.Resampler.aggregate, squeeze_self=True)(self, resample_kwargs, func, *args, **kwargs)