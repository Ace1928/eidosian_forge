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
def round_trip_localpath(writer, reader, path: str | None=None):
    """
    Write an object to file specified by a py.path LocalPath and read it back.

    Parameters
    ----------
    writer : callable bound to pandas object
        IO writing function (e.g. DataFrame.to_csv )
    reader : callable
        IO reading function (e.g. pd.read_csv )
    path : str, default None
        The path where the object is written and then read.

    Returns
    -------
    pandas object
        The original object that was serialized and then re-read.
    """
    import pytest
    LocalPath = pytest.importorskip('py.path').local
    if path is None:
        path = '___localpath___'
    with ensure_clean(path) as path:
        writer(LocalPath(path))
        obj = reader(LocalPath(path))
    return obj