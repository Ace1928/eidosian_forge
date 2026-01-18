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
def test_fragments_parquet_row_groups_reconstruct(tempdir, dataset_reader, pickle_module):
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=2)
    fragment = list(dataset.get_fragments())[0]
    parquet_format = fragment.format
    row_group_fragments = list(fragment.split_by_row_group())
    pickled_fragment = pickle_module.loads(pickle_module.dumps(fragment))
    assert dataset_reader.to_table(pickled_fragment) == dataset_reader.to_table(fragment)
    new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression, row_groups=[0])
    result = dataset_reader.to_table(new_fragment)
    assert result.equals(dataset_reader.to_table(row_group_fragments[0]))
    new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression, row_groups={1})
    result = dataset_reader.to_table(new_fragment, schema=table.schema, columns=['f1', 'part'], filter=ds.field('f1') < 3)
    assert result.column_names == ['f1', 'part']
    assert len(result) == 1
    new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression, row_groups={2})
    with pytest.raises(IndexError, match='references row group 2'):
        dataset_reader.to_table(new_fragment)