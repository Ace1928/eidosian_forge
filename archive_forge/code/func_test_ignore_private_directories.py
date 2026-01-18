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
@pytest.mark.pandas
@pytest.mark.parametrize('dir_prefix', ['_', '.'])
def test_ignore_private_directories(tempdir, dir_prefix):
    dirpath = tempdir / guid()
    dirpath.mkdir()
    paths = _make_example_multifile_dataset(dirpath, nfiles=10, file_nrows=5)
    (dirpath / '{}staging'.format(dir_prefix)).mkdir()
    dataset = pq.ParquetDataset(dirpath)
    _assert_dataset_paths(dataset, paths)