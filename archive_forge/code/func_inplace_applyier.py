import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
def inplace_applyier(grp, *func_args, **func_kwargs):
    return key(grp, *inplace_args, *func_args, **func_kwargs)