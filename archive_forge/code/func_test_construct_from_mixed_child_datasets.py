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
def test_construct_from_mixed_child_datasets(mockfs):
    a = ds.dataset(['subdir/1/xxx/file0.parquet', 'subdir/2/yyy/file1.parquet'], filesystem=mockfs)
    b = ds.dataset('subdir', filesystem=mockfs)
    dataset = ds.dataset([a, b])
    assert isinstance(dataset, ds.UnionDataset)
    assert len(list(dataset.get_fragments())) == 4
    table = dataset.to_table()
    assert len(table) == 20
    assert table.num_columns == 5
    assert len(dataset.children) == 2
    for child in dataset.children:
        assert child.files == ['subdir/1/xxx/file0.parquet', 'subdir/2/yyy/file1.parquet']