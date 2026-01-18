import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
def find_common_type_cat(types):
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : array-like
        Array of dtypes.

    Returns
    -------
    pandas.core.dtypes.dtypes.ExtensionDtype or
    np.dtype or
    None
        `dtype` that is common for all passed `types`.
    """
    if all((isinstance(t, pandas.CategoricalDtype) for t in types)):
        if all((t.ordered for t in types)):
            categories = np.sort(np.unique([c for t in types for c in t.categories]))
            return pandas.CategoricalDtype(categories, ordered=True)
        return union_categoricals([pandas.Categorical([], dtype=t) for t in types], sort_categories=all((t.ordered for t in types))).dtype
    else:
        return find_common_type(list(types))