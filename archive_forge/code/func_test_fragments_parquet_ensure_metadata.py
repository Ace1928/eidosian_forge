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
def test_fragments_parquet_ensure_metadata(tempdir, open_logging_fs, pickle_module):
    fs, assert_opens = open_logging_fs
    _, dataset = _create_dataset_for_fragments(tempdir, chunk_size=2, filesystem=fs)
    fragment = list(dataset.get_fragments())[0]
    with assert_opens([fragment.path]):
        fragment.ensure_complete_metadata()
    assert fragment.row_groups == [0, 1]
    with assert_opens([]):
        fragment.ensure_complete_metadata()
    assert isinstance(fragment.metadata, pq.FileMetaData)
    new_fragment = fragment.format.make_fragment(fragment.path, fragment.filesystem, row_groups=[0, 1])
    assert new_fragment.row_groups == fragment.row_groups
    new_fragment.ensure_complete_metadata()
    row_group = new_fragment.row_groups[0]
    assert row_group.id == 0
    assert row_group.num_rows == 2
    assert row_group.statistics is not None
    pickled_fragment = pickle_module.loads(pickle_module.dumps(new_fragment))
    with assert_opens([fragment.path]):
        assert pickled_fragment.row_groups == [0, 1]
        row_group = pickled_fragment.row_groups[0]
        assert row_group.id == 0
        assert row_group.statistics is not None