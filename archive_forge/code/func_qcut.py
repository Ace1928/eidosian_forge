from __future__ import annotations
import warnings
from typing import Hashable, Iterable, Mapping, Optional, Union
import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import ArrayLike, DtypeBackend, Scalar, npt
from pandas.core.dtypes.common import is_list_like
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import enable_logging
from modin.pandas.io import to_pandas
from modin.utils import _inherit_docstrings
from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series
@_inherit_docstrings(pandas.qcut, apilink='pandas.qcut')
@enable_logging
def qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise'):
    """
    Quantile-based discretization function.
    """
    kwargs = {'labels': labels, 'retbins': retbins, 'precision': precision, 'duplicates': duplicates}
    if not isinstance(x, Series):
        return pandas.qcut(x, q, **kwargs)
    return x._qcut(q, **kwargs)