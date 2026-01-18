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
def test_fragments_parquet_subset_filter(tempdir, open_logging_fs, dataset_reader):
    fs, assert_opens = open_logging_fs
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=1, filesystem=fs)
    fragment = list(dataset.get_fragments())[0]
    subfrag = fragment.subset(ds.field('f1') >= 1)
    with assert_opens([]):
        assert subfrag.num_row_groups == 3
        assert len(subfrag.row_groups) == 3
        assert subfrag.row_groups[0].statistics is not None
    result = dataset_reader.to_table(subfrag)
    assert result.to_pydict() == {'f1': [1, 2, 3], 'f2': [1, 1, 1]}
    subfrag = fragment.subset(ds.field('f1') > 5)
    assert subfrag.num_row_groups == 0
    assert subfrag.row_groups == []
    result = dataset_reader.to_table(subfrag, schema=dataset.schema)
    assert result.num_rows == 0
    assert result.equals(table[:0])
    subfrag = fragment.subset(ds.field('part') == 'a', schema=dataset.schema)
    assert subfrag.num_row_groups == 4