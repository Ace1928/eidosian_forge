import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def test_dataset_read_dictionary(tempdir):
    path = tempdir / 'ARROW-3325-dataset'
    t1 = pa.table([[util.rands(10) for i in range(5)] * 10], names=['f0'])
    t2 = pa.table([[util.rands(10) for i in range(5)] * 10], names=['f0'])
    pq.write_to_dataset(t1, root_path=str(path))
    pq.write_to_dataset(t2, root_path=str(path))
    result = pq.ParquetDataset(path, read_dictionary=['f0']).read()
    ex_chunks = [t1[0].chunk(0).dictionary_encode(), t2[0].chunk(0).dictionary_encode()]
    assert result[0].num_chunks == 2
    c0, c1 = (result[0].chunk(0), result[0].chunk(1))
    if c0.equals(ex_chunks[0]):
        assert c1.equals(ex_chunks[1])
    else:
        assert c0.equals(ex_chunks[1])
        assert c1.equals(ex_chunks[0])