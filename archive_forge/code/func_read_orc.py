from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Literal
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import DataFrame, Scalar
from dask.dataframe.io.io import from_map
from dask.dataframe.io.orc.utils import ORCEngine
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply
@dataframe_creation_dispatch.register_inplace('pandas')
def read_orc(path, engine='pyarrow', columns=None, index=None, split_stripes=1, aggregate_files=None, storage_options=None):
    """Read dataframe from ORC file(s)

    Parameters
    ----------
    path: str or list(str)
        Location of file(s), which can be a full URL with protocol
        specifier, and may include glob character if a single string.
    engine: 'pyarrow' or ORCEngine
        Backend ORC engine to use for I/O. Default is "pyarrow".
    columns: None or list(str)
        Columns to load. If None, loads all.
    index: str
        Column name to set as index.
    split_stripes: int or False
        Maximum number of ORC stripes to include in each output-DataFrame
        partition. Use False to specify a 1-to-1 mapping between files
        and partitions. Default is 1.
    aggregate_files : bool, default False
        Whether distinct file paths may be aggregated into the same output
        partition. A setting of True means that any two file paths may be
        aggregated into the same output partition, while False means that
        inter-file aggregation is prohibited.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.

    Returns
    -------
    Dask.DataFrame (even if there is only one column)

    Examples
    --------
    >>> df = dd.read_orc('https://github.com/apache/orc/raw/'
    ...                  'master/examples/demo-11-zlib.orc')  # doctest: +SKIP
    """
    engine = _get_engine(engine)
    storage_options = storage_options or {}
    fs, fs_token, paths = get_fs_token_paths(path, mode='rb', storage_options=storage_options)
    parts, schema, meta = engine.read_metadata(fs, paths, columns, index, split_stripes, aggregate_files)
    return from_map(ORCFunctionWrapper(fs, columns, schema, engine, index), parts, meta=meta, divisions=[None] * (len(parts) + 1), label='read-orc', token=tokenize(fs_token, path, columns), enforce_metadata=False)