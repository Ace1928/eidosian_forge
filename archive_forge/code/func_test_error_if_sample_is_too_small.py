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
def test_error_if_sample_is_too_small():
    text = 'AAAAA,BBBBB,CCCCC,DDDDD,EEEEE\n1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15'
    with filetext(text) as fn:
        sample = 20
        with pytest.raises(ValueError):
            dd.read_csv(fn, sample=sample)
        assert_eq(dd.read_csv(fn, sample=sample, header=None), pd.read_csv(fn, header=None))
    skiptext = '# skip\n# these\n# lines\n'
    text = skiptext + text
    with filetext(text) as fn:
        sample = 20 + len(skiptext)
        with pytest.raises(ValueError):
            dd.read_csv(fn, sample=sample, skiprows=3)
        assert_eq(dd.read_csv(fn, sample=sample, header=None, skiprows=3), pd.read_csv(fn, header=None, skiprows=3))