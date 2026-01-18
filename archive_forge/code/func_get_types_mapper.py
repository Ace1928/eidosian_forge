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
@staticmethod
def get_types_mapper(dtype_backend):
    """
        Get types mapper that would be used in read_parquet/read_feather.

        Parameters
        ----------
        dtype_backend : {"numpy_nullable", "pyarrow", lib.no_default}

        Returns
        -------
        dict
        """
    to_pandas_kwargs = {}
    if dtype_backend == 'numpy_nullable':
        from pandas.io._util import _arrow_dtype_mapping
        mapping = _arrow_dtype_mapping()
        to_pandas_kwargs['types_mapper'] = mapping.get
    elif dtype_backend == 'pyarrow':
        to_pandas_kwargs['types_mapper'] = pandas.ArrowDtype
    return to_pandas_kwargs