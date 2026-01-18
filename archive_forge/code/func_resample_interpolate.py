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
@doc_utils.doc_resample_fillna(method='specified interpolation', refer_to='interpolate', params='\n        method : str\n        axis : {0, 1}\n        limit : int\n        inplace : {False}\n            This parameter serves the compatibility purpose. Always has to be False.\n        limit_direction : {"forward", "backward", "both"}\n        limit_area : {None, "inside", "outside"}\n        downcast : str, optional\n        **kwargs : dict\n        ', overwrite_template_params=True)
def resample_interpolate(self, resample_kwargs, method, axis, limit, inplace, limit_direction, limit_area, downcast, **kwargs):
    return ResampleDefault.register(pandas.core.resample.Resampler.interpolate)(self, resample_kwargs, method, axis=axis, limit=limit, inplace=inplace, limit_direction=limit_direction, limit_area=limit_area, downcast=downcast, **kwargs)