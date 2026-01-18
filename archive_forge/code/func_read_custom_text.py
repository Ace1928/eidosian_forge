from __future__ import annotations
import inspect
import pathlib
import pickle
from typing import IO, AnyStr, Callable, Iterator, Literal, Optional, Union
import pandas
import pandas._libs.lib as lib
from pandas._typing import CompressionOptions, DtypeArg, DtypeBackend, StorageOptions
from modin.core.storage_formats import BaseQueryCompiler
from modin.utils import expanduser_path_arg
from . import DataFrame
@expanduser_path_arg('filepath_or_buffer')
def read_custom_text(filepath_or_buffer, columns, custom_parser, compression='infer', nrows: Optional[int]=None, is_quoting=True):
    """
    Load custom text data from file.

    Parameters
    ----------
    filepath_or_buffer : str
        File path where the custom text data will be loaded from.
    columns : list or callable(file-like object, \\*\\*kwargs) -> list
        Column names of list type or callable that create column names from opened file
        and passed `kwargs`.
    custom_parser : callable(file-like object, \\*\\*kwargs) -> pandas.DataFrame
        Function that takes as input a part of the `filepath_or_buffer` file loaded into
        memory in file-like object form.
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default: 'infer'
        If 'infer' and 'path_or_url' is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        compression). If 'infer' and 'path_or_url' is not path-like, then use
        None (= no decompression).
    nrows : int, optional
        Amount of rows to read.
    is_quoting : bool, default: True
        Whether or not to consider quotes.

    Returns
    -------
    modin.DataFrame
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    return DataFrame(query_compiler=FactoryDispatcher.read_custom_text(**kwargs))