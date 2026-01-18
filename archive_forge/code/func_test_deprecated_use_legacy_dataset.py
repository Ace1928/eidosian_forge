from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
@pytest.mark.dataset
def test_deprecated_use_legacy_dataset(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'deprecate_legacy'
    msg = "Passing 'use_legacy_dataset'"
    with pytest.warns(FutureWarning, match=msg):
        pq.write_to_dataset(table, path, use_legacy_dataset=False)
    pq.write_to_dataset(table, path)
    with pytest.warns(FutureWarning, match=msg):
        pq.read_table(path, use_legacy_dataset=False)
    with pytest.warns(FutureWarning, match=msg):
        pq.ParquetDataset(path, use_legacy_dataset=False)