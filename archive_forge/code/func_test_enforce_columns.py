from __future__ import annotations
import gzip
import os
import warnings
from io import BytesIO, StringIO
from unittest import mock
import pytest
import fsspec
from fsspec.compression import compr
from packaging.version import Version
from tlz import partition_all, valmap
import dask
from dask.base import compute_as_if_collection
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.io.csv import (
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import (
from dask.layers import DataFrameIOLayer
from dask.utils import filetext, filetexts, tmpdir, tmpfile
from dask.utils_test import hlg_layer
@pytest.mark.parametrize('reader,blocks', [(pd.read_csv, csv_blocks), (pd.read_table, tsv_blocks)])
def test_enforce_columns(reader, blocks):
    blocks = [blocks[0], [blocks[1][0].replace(b'a', b'A'), blocks[1][1]]]
    head = reader(BytesIO(blocks[0][0]), header=0)
    header = blocks[0][0].split(b'\n')[0] + b'\n'
    with pytest.raises(ValueError):
        dfs = text_blocks_to_pandas(reader, blocks, header, head, {}, enforce=True)
        dask.compute(*dfs, scheduler='sync')