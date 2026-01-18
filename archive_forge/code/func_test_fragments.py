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
def test_fragments(tempdir, dataset_reader):
    table, dataset = _create_dataset_for_fragments(tempdir)
    fragments = list(dataset.get_fragments())
    assert len(fragments) == 2
    f = fragments[0]
    physical_names = ['f1', 'f2']
    assert f.physical_schema.names == physical_names
    assert f.format.inspect(f.path, f.filesystem) == f.physical_schema
    assert f.partition_expression.equals(ds.field('part') == 'a')
    result = dataset_reader.to_table(f)
    assert result.column_names == physical_names
    assert result.equals(table.remove_column(2).slice(0, 4))
    result = dataset_reader.to_table(f, schema=dataset.schema)
    assert result.column_names == ['f1', 'f2', 'part']
    assert result.equals(table.slice(0, 4))
    assert f.physical_schema == result.schema.remove(2)
    result = dataset_reader.to_table(f, schema=dataset.schema, filter=ds.field('f1') < 2)
    assert result.column_names == ['f1', 'f2', 'part']