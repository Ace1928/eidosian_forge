from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
def check_compression(engine, filename, compression):
    if engine == 'fastparquet':
        pf = fastparquet.ParquetFile(filename)
        md = pf.fmd.row_groups[0].columns[0].meta_data
        if compression is None:
            assert md.total_compressed_size == md.total_uncompressed_size
        else:
            assert md.total_compressed_size != md.total_uncompressed_size
    else:
        assert engine == 'pyarrow'
        metadata = pa.parquet.read_metadata(os.path.join(filename, '_metadata'))
        names = metadata.schema.names
        for i in range(metadata.num_row_groups):
            row_group = metadata.row_group(i)
            for j in range(len(names)):
                column = row_group.column(j)
                if compression is None:
                    assert column.total_compressed_size == column.total_uncompressed_size
                else:
                    compress_expect = compression
                    if compression == 'default':
                        compress_expect = 'snappy'
                    assert compress_expect.lower() == column.compression.lower()
                    assert column.total_compressed_size != column.total_uncompressed_size