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
def test_column_names_encoding(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')
    with open(path, 'wb') as sink:
        sink.write(b'\xe9,b\nun,\xe9l\xe9phant')
    expected_schema = pa.schema([('é', pa.string()), ('b', pa.string())])
    expected_table = pa.table({'é': ['un'], 'b': ['éléphant']}, schema=expected_schema)
    dataset = ds.dataset(path, format='csv', schema=expected_schema)
    with pytest.raises(pyarrow.lib.ArrowInvalid, match='invalid UTF8'):
        dataset_reader.to_table(dataset)
    read_options = pa.csv.ReadOptions(encoding='latin-1')
    file_format = ds.CsvFileFormat(read_options=read_options)
    dataset_transcoded = ds.dataset(path, format=file_format)
    assert dataset_transcoded.schema.equals(expected_schema)
    assert dataset_transcoded.to_table().equals(expected_table)