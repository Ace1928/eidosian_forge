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
def test_file_format_pickling(pickle_module):
    formats = [ds.IpcFileFormat(), ds.CsvFileFormat(), ds.CsvFileFormat(pa.csv.ParseOptions(delimiter='\t', ignore_empty_lines=True)), ds.CsvFileFormat(read_options=pa.csv.ReadOptions(skip_rows=3, column_names=['foo'])), ds.CsvFileFormat(read_options=pa.csv.ReadOptions(skip_rows=3, block_size=2 ** 20)), ds.JsonFileFormat(), ds.JsonFileFormat(parse_options=pa.json.ParseOptions(newlines_in_values=True, unexpected_field_behavior='ignore')), ds.JsonFileFormat(read_options=pa.json.ReadOptions(use_threads=False, block_size=14))]
    try:
        formats.append(ds.OrcFileFormat())
    except ImportError:
        pass
    if pq is not None:
        formats.extend([ds.ParquetFileFormat(), ds.ParquetFileFormat(dictionary_columns={'a'}), ds.ParquetFileFormat(use_buffered_stream=True), ds.ParquetFileFormat(use_buffered_stream=True, buffer_size=4096, thrift_string_size_limit=123, thrift_container_size_limit=456)])
    for file_format in formats:
        assert pickle_module.loads(pickle_module.dumps(file_format)) == file_format