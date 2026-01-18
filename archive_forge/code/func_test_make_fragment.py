import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parquet
def test_make_fragment(multisourcefs):
    parquet_format = ds.ParquetFileFormat()
    dataset = ds.dataset('/plain', filesystem=multisourcefs, format=parquet_format)
    for path in dataset.files:
        fragment = parquet_format.make_fragment(path, multisourcefs)
        assert fragment.row_groups == [0]
        row_group_fragment = parquet_format.make_fragment(path, multisourcefs, row_groups=[0])
        for f in [fragment, row_group_fragment]:
            assert isinstance(f, ds.ParquetFileFragment)
            assert f.path == path
            assert isinstance(f.filesystem, type(multisourcefs))
        assert row_group_fragment.row_groups == [0]