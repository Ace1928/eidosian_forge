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
def test_fragment_scan_options_pickling(pickle_module):
    options = [ds.CsvFragmentScanOptions(), ds.CsvFragmentScanOptions(convert_options=pa.csv.ConvertOptions(strings_can_be_null=True)), ds.CsvFragmentScanOptions(read_options=pa.csv.ReadOptions(block_size=2 ** 16)), ds.JsonFragmentScanOptions(), ds.JsonFragmentScanOptions(pa.json.ParseOptions(newlines_in_values=False, unexpected_field_behavior='error')), ds.JsonFragmentScanOptions(read_options=pa.json.ReadOptions(use_threads=True, block_size=512))]
    if pq is not None:
        options.extend([ds.ParquetFragmentScanOptions(buffer_size=4096), ds.ParquetFragmentScanOptions(pre_buffer=True)])
    for option in options:
        assert pickle_module.loads(pickle_module.dumps(option)) == option