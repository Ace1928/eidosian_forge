from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from fsspec.core import expand_paths_if_needed, get_fs_token_paths, stringify_path
from fsspec.spec import AbstractFileSystem
from dask import config
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key, parse_bytes
@classmethod
def aggregate_metadata(cls, meta_list, fs, out_path):
    """
        Aggregate a list of metadata objects and optionally
        write out the final result as a _metadata file.

        Parameters
        ----------
        meta_list: list
            List of metadata objects to be aggregated into a single
            metadata object, and optionally written to disk. The
            specific element type can be engine specific.
        fs: FileSystem
        out_path: str or None
            Directory to write the final _metadata file. If None
            is specified, the aggregated metadata will be returned,
            and nothing will be written to disk.

        Returns
        -------
        If out_path is None, an aggregate metadata object is returned.
        Otherwise, None is returned.
        """
    raise NotImplementedError()