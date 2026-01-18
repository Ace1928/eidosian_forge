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
@pytest.mark.pandas
@pytest.mark.parametrize('compression', ['lz4', 'zstd', 'brotli'])
def test_feather_format_compressed(tempdir, compression, dataset_reader):
    table = pa.table({'a': pa.array([0] * 300, type='int8'), 'b': pa.array([0.1, 0.2, 0.3] * 100, type='float64')})
    if not pa.Codec.is_available(compression):
        pytest.skip()
    basedir = tempdir / 'feather_dataset_compressed'
    basedir.mkdir()
    file_format = ds.IpcFileFormat()
    uncompressed_basedir = tempdir / 'feather_dataset_uncompressed'
    uncompressed_basedir.mkdir()
    ds.write_dataset(table, str(uncompressed_basedir / 'data.arrow'), format=file_format, file_options=file_format.make_write_options(compression=None))
    if compression == 'brotli':
        with pytest.raises(ValueError, match='Compression type'):
            write_options = file_format.make_write_options(compression=compression)
        with pytest.raises(ValueError, match='Compression type'):
            codec = pa.Codec(compression)
            write_options = file_format.make_write_options(compression=codec)
        return
    write_options = file_format.make_write_options(compression=compression)
    ds.write_dataset(table, str(basedir / 'data.arrow'), format=file_format, file_options=write_options)
    dataset = ds.dataset(basedir, format=ds.IpcFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)
    compressed_file = basedir / 'data.arrow' / 'part-0.arrow'
    compressed_size = compressed_file.stat().st_size
    uncompressed_file = uncompressed_basedir / 'data.arrow' / 'part-0.arrow'
    uncompressed_size = uncompressed_file.stat().st_size
    assert compressed_size < uncompressed_size