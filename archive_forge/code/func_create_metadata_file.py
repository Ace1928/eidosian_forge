from __future__ import annotations
import contextlib
import math
import warnings
from typing import Literal
import pandas as pd
import tlz as toolz
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
import dask
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import DataFrame, Scalar
from dask.dataframe.io.io import from_map
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import DataFrameIOFunction, _is_local_fs
from dask.dataframe.methods import concat
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import apply, import_required, natural_sort_key, parse_bytes
def create_metadata_file(paths, root_dir=None, out_dir=None, engine='pyarrow', storage_options=None, split_every=32, compute=True, compute_kwargs=None, fs=None):
    """Construct a global _metadata file from a list of parquet files.

    Dask's read_parquet function is designed to leverage a global
    _metadata file whenever one is available.  The to_parquet
    function will generate this file automatically by default, but it
    may not exist if the dataset was generated outside of Dask.  This
    utility provides a mechanism to generate a _metadata file from a
    list of existing parquet files.

    Parameters
    ----------
    paths : list(string)
        List of files to collect footer metadata from.
    root_dir : string, optional
        Root directory of dataset.  The `file_path` fields in the new
        _metadata file will relative to this directory.  If None, a common
        root directory will be inferred.
    out_dir : string or False, optional
        Directory location to write the final _metadata file.  By default,
        this will be set to `root_dir`.  If False is specified, the global
        metadata will be returned as an in-memory object (and will not be
        written to disk).
    engine : str or Engine, default 'pyarrow'
        Parquet Engine to use. Only 'pyarrow' is supported if a string
        is passed.
    storage_options : dict, optional
        Key/value pairs to be passed on to the file-system backend, if any.
    split_every : int, optional
        The final metadata object that is written to _metadata can be much
        smaller than the list of footer metadata. In order to avoid the
        aggregation of all metadata within a single task, a tree reduction
        is used.  This argument specifies the maximum number of metadata
        inputs to be handled by any one task in the tree. Defaults to 32.
    compute : bool, optional
        If True (default) then the result is computed immediately. If False
        then a ``dask.delayed`` object is returned for future computation.
    compute_kwargs : dict, optional
        Options to be passed in to the compute method
    fs : fsspec object, optional
        File-system instance to use for file handling. If prefixes have
        been removed from the elements of ``paths`` before calling this
        function, an ``fs`` argument must be provided to ensure correct
        behavior on remote file systems ("naked" paths cannot be used
        to infer file-system information).
    """
    if isinstance(engine, str):
        engine = get_engine(engine)
        if engine is not _ENGINES.get('pyarrow'):
            raise ValueError('fastparquet is not a supported engine for create_metadata_file.Please install pyarrow.')
    if fs is None:
        fs, _, paths = get_fs_token_paths(paths, mode='rb', storage_options=storage_options)
    ap_kwargs = {'root': root_dir} if root_dir else {}
    paths, root_dir, fns = _sort_and_analyze_paths(paths, fs, **ap_kwargs)
    out_dir = root_dir if out_dir is None else out_dir
    dsk = {}
    name = 'gen-metadata-' + tokenize(paths, fs)
    collect_name = 'collect-' + name
    agg_name = 'agg-' + name
    for p, (fn, path) in enumerate(zip(fns, paths)):
        key = (collect_name, p, 0)
        dsk[key] = (engine.collect_file_metadata, path, fs, fn)
    parts = len(paths)
    widths = [parts]
    while parts > 1:
        parts = math.ceil(parts / split_every)
        widths.append(parts)
    height = len(widths)
    for depth in range(1, height):
        for group in range(widths[depth]):
            p_max = widths[depth - 1]
            lstart = split_every * group
            lstop = min(lstart + split_every, p_max)
            dep_task_name = collect_name if depth == 1 else agg_name
            node_list = [(dep_task_name, p, depth - 1) for p in range(lstart, lstop)]
            if depth == height - 1:
                assert group == 0
                dsk[name] = (engine.aggregate_metadata, node_list, fs, out_dir)
            else:
                dsk[agg_name, group, depth] = (engine.aggregate_metadata, node_list, None, None)
    if len(paths) == 1:
        dsk[name] = (engine.aggregate_metadata, [(collect_name, 0, 0)], fs, out_dir)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[])
    out = Delayed(name, graph)
    if compute:
        if compute_kwargs is None:
            compute_kwargs = dict()
        out = out.compute(**compute_kwargs)
    return out