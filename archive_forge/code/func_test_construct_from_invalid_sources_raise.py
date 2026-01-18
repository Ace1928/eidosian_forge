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
def test_construct_from_invalid_sources_raise(multisourcefs):
    child1 = ds.FileSystemDatasetFactory(multisourcefs, fs.FileSelector('/plain'), format=ds.ParquetFileFormat())
    child2 = ds.FileSystemDatasetFactory(multisourcefs, fs.FileSelector('/schema'), format=ds.ParquetFileFormat())
    batch1 = pa.RecordBatch.from_arrays([pa.array(range(10))], names=['a'])
    batch2 = pa.RecordBatch.from_arrays([pa.array(range(10))], names=['b'])
    with pytest.raises(TypeError, match='Expected.*FileSystemDatasetFactory'):
        ds.dataset([child1, child2])
    expected = 'Expected a list of path-like or dataset objects, or a list of batches or tables. The given list contains the following types: int'
    with pytest.raises(TypeError, match=expected):
        ds.dataset([1, 2, 3])
    expected = 'Expected a path-like, list of path-likes or a list of Datasets instead of the given type: NoneType'
    with pytest.raises(TypeError, match=expected):
        ds.dataset(None)
    expected = 'Expected a path-like, list of path-likes or a list of Datasets instead of the given type: generator'
    with pytest.raises(TypeError, match=expected):
        ds.dataset((batch1 for _ in range(3)))
    expected = 'Must provide schema to construct in-memory dataset from an empty list'
    with pytest.raises(ValueError, match=expected):
        ds.InMemoryDataset([])
    expected = 'Item has schema\nb: int64\nwhich does not match expected schema\na: int64'
    with pytest.raises(TypeError, match=expected):
        ds.dataset([batch1, batch2])
    expected = 'Expected a list of path-like or dataset objects, or a list of batches or tables. The given list contains the following types:'
    with pytest.raises(TypeError, match=expected):
        ds.dataset([batch1, 0])
    expected = 'Expected a list of tables or batches. The given list contains a int'
    with pytest.raises(TypeError, match=expected):
        ds.InMemoryDataset([batch1, 0])