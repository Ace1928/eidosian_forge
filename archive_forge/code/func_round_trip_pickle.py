from __future__ import annotations
import gzip
import io
import pathlib
import tarfile
from typing import (
import uuid
import zipfile
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas as pd
from pandas._testing.contexts import ensure_clean
def round_trip_pickle(obj: Any, path: FilePath | ReadPickleBuffer | None=None) -> DataFrame | Series:
    """
    Pickle an object and then read it again.

    Parameters
    ----------
    obj : any object
        The object to pickle and then re-read.
    path : str, path object or file-like object, default None
        The path where the pickled object is written and then read.

    Returns
    -------
    pandas object
        The original object that was pickled and then re-read.
    """
    _path = path
    if _path is None:
        _path = f'__{uuid.uuid4()}__.pickle'
    with ensure_clean(_path) as temp_path:
        pd.to_pickle(obj, temp_path)
        return pd.read_pickle(temp_path)