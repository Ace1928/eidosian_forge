import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from fsspec import AbstractFileSystem
from triad import Schema, assert_or_throw
from triad.collections.schema import SchemaError
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.io import url_to_fs
from triad.utils.pyarrow import pa_batch_to_dicts
from .api import as_fugue_df, get_column_names, normalize_column_names, rename
from .dataframe import DataFrame, LocalBoundedDataFrame
def serialize_df(df: Optional[DataFrame], threshold: int=-1, file_path: Optional[str]=None) -> Optional[bytes]:
    """Serialize input dataframe to base64 string or to file
    if it's larger than threshold

    :param df: input DataFrame
    :param threshold: file byte size threshold, defaults to -1
    :param file_path: file path to store the data (used only if the serialized data
      is larger than ``threshold``), defaults to None
    :raises InvalidOperationError: if file is large but ``file_path`` is not provided
    :return: a pickled blob either containing the data or the file path
    """
    if df is None:
        return None
    data = pickle.dumps(df.as_local_bounded())
    size = len(data)
    if threshold < 0 or size <= threshold:
        return data
    else:
        if file_path is None:
            raise InvalidOperationError('file_path is not provided')
        fs, path = url_to_fs(file_path)
        with fs.open(path, 'wb') as f:
            f.write(data)
        return pickle.dumps(file_path)