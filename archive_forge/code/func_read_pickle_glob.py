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
def read_pickle_glob(filepath_or_buffer, compression: Optional[str]='infer', storage_options: StorageOptions=None):
    """
    Load pickled pandas object from files.

    This experimental feature provides parallel reading from multiple pickle files which are
    defined by glob pattern. The files must contain parts of one dataframe, which can be
    obtained, for example, by `DataFrame.modin.to_pickle_glob` function.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        File path, URL, or buffer where the pickled object will be loaded from.
        Accept URL. URL is not limited to S3 and GCS.
    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default: 'infer'
        If 'infer' and 'path_or_url' is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        compression) If 'infer' and 'path_or_url' is not path-like, then use
        None (= no decompression).
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc., if using a URL that will be parsed by
        fsspec, e.g., starting "s3://", "gcs://". An error will be raised if providing
        this argument with a non-fsspec URL. See the fsspec and backend storage
        implementation docs for the set of allowed keys and values.

    Returns
    -------
    unpickled : same type as object stored in file

    Notes
    -----
    The number of partitions is equal to the number of input files.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    return DataFrame(query_compiler=FactoryDispatcher.read_pickle_glob(**kwargs))