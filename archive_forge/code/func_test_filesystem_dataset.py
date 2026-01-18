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
def test_filesystem_dataset(mockfs):
    schema = pa.schema([pa.field('const', pa.int64())])
    file_format = ds.ParquetFileFormat()
    paths = ['subdir/1/xxx/file0.parquet', 'subdir/2/yyy/file1.parquet']
    partitions = [ds.field('part') == x for x in range(1, 3)]
    fragments = [file_format.make_fragment(path, mockfs, part) for path, part in zip(paths, partitions)]
    root_partition = ds.field('level') == ds.scalar(1337)
    dataset_from_fragments = ds.FileSystemDataset(fragments, schema=schema, format=file_format, filesystem=mockfs, root_partition=root_partition)
    dataset_from_paths = ds.FileSystemDataset.from_paths(paths, schema=schema, format=file_format, filesystem=mockfs, partitions=partitions, root_partition=root_partition)
    for dataset in [dataset_from_fragments, dataset_from_paths]:
        assert isinstance(dataset, ds.FileSystemDataset)
        assert isinstance(dataset.format, ds.ParquetFileFormat)
        assert dataset.partition_expression.equals(root_partition)
        assert set(dataset.files) == set(paths)
        fragments = list(dataset.get_fragments())
        for fragment, partition, path in zip(fragments, partitions, paths):
            assert fragment.partition_expression.equals(partition)
            assert fragment.path == path
            assert isinstance(fragment.format, ds.ParquetFileFormat)
            assert isinstance(fragment, ds.ParquetFileFragment)
            assert fragment.row_groups == [0]
            assert fragment.num_row_groups == 1
            row_group_fragments = list(fragment.split_by_row_group())
            assert fragment.num_row_groups == len(row_group_fragments) == 1
            assert isinstance(row_group_fragments[0], ds.ParquetFileFragment)
            assert row_group_fragments[0].path == path
            assert row_group_fragments[0].row_groups == [0]
            assert row_group_fragments[0].num_row_groups == 1
        fragments = list(dataset.get_fragments(filter=ds.field('const') == 0))
        assert len(fragments) == 2
    dataset = ds.FileSystemDataset(fragments, schema=schema, format=file_format, filesystem=mockfs)
    assert dataset.partition_expression.equals(ds.scalar(True))
    dataset = ds.FileSystemDataset.from_paths(paths, schema=schema, format=file_format, filesystem=mockfs)
    assert dataset.partition_expression.equals(ds.scalar(True))
    for fragment in dataset.get_fragments():
        assert fragment.partition_expression.equals(ds.scalar(True))
    with pytest.raises(TypeError, match='incorrect type'):
        ds.FileSystemDataset(fragments, file_format, schema)
    with pytest.raises(TypeError, match='incorrect type'):
        ds.FileSystemDataset(fragments, schema=schema, format=file_format, root_partition=1)
    with pytest.raises(TypeError, match='incorrect type'):
        ds.FileSystemDataset.from_paths(fragments, format=file_format)