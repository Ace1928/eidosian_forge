import io
import json
import warnings
from .core import url_to_fs
from .utils import merge_offset_ranges
def open_parquet_file(path, mode='rb', fs=None, metadata=None, columns=None, row_groups=None, storage_options=None, strict=False, engine='auto', max_gap=64000, max_block=256000000, footer_sample_size=1000000, **kwargs):
    """
    Return a file-like object for a single Parquet file.

    The specified parquet `engine` will be used to parse the
    footer metadata, and determine the required byte ranges
    from the file. The target path will then be opened with
    the "parts" (`KnownPartsOfAFile`) caching strategy.

    Note that this method is intended for usage with remote
    file systems, and is unlikely to improve parquet-read
    performance on local file systems.

    Parameters
    ----------
    path: str
        Target file path.
    mode: str, optional
        Mode option to be passed through to `fs.open`. Default is "rb".
    metadata: Any, optional
        Parquet metadata object. Object type must be supported
        by the backend parquet engine. For now, only the "fastparquet"
        engine supports an explicit `ParquetFile` metadata object.
        If a metadata object is supplied, the remote footer metadata
        will not need to be transferred into local memory.
    fs: AbstractFileSystem, optional
        Filesystem object to use for opening the file. If nothing is
        specified, an `AbstractFileSystem` object will be inferred.
    engine : str, default "auto"
        Parquet engine to use for metadata parsing. Allowed options
        include "fastparquet", "pyarrow", and "auto". The specified
        engine must be installed in the current environment. If
        "auto" is specified, and both engines are installed,
        "fastparquet" will take precedence over "pyarrow".
    columns: list, optional
        List of all column names that may be read from the file.
    row_groups : list, optional
        List of all row-groups that may be read from the file. This
        may be a list of row-group indices (integers), or it may be
        a list of `RowGroup` metadata objects (if the "fastparquet"
        engine is used).
    storage_options : dict, optional
        Used to generate an `AbstractFileSystem` object if `fs` was
        not specified.
    strict : bool, optional
        Whether the resulting `KnownPartsOfAFile` cache should
        fetch reads that go beyond a known byte-range boundary.
        If `False` (the default), any read that ends outside a
        known part will be zero padded. Note that using
        `strict=True` may be useful for debugging.
    max_gap : int, optional
        Neighboring byte ranges will only be merged when their
        inter-range gap is <= `max_gap`. Default is 64KB.
    max_block : int, optional
        Neighboring byte ranges will only be merged when the size of
        the aggregated range is <= `max_block`. Default is 256MB.
    footer_sample_size : int, optional
        Number of bytes to read from the end of the path to look
        for the footer metadata. If the sampled bytes do not contain
        the footer, a second read request will be required, and
        performance will suffer. Default is 1MB.
    **kwargs :
        Optional key-word arguments to pass to `fs.open`
    """
    if fs is None:
        fs = url_to_fs(path, **storage_options or {})[0]
    if columns is not None and len(columns) == 0:
        return fs.open(path, mode=mode)
    engine = _set_engine(engine)
    data = _get_parquet_byte_ranges([path], fs, metadata=metadata, columns=columns, row_groups=row_groups, engine=engine, max_gap=max_gap, max_block=max_block, footer_sample_size=footer_sample_size)
    fn = next(iter(data)) if data else path
    options = kwargs.pop('cache_options', {}).copy()
    return fs.open(fn, mode=mode, cache_type='parts', cache_options={**options, 'data': data.get(fn, {}), 'strict': strict}, **kwargs)