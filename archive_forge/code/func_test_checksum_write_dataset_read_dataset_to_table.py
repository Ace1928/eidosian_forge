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
def test_checksum_write_dataset_read_dataset_to_table(tempdir):
    """Check that checksum verification works for datasets created with
    ds.write_dataset and read with ds.dataset.to_table"""
    table_orig = pa.table({'a': [1, 2, 3, 4]})
    pq_write_format = pa.dataset.ParquetFileFormat()
    write_options = pq_write_format.make_write_options(write_page_checksum=True)
    original_dir_path = tempdir / 'correct_dir'
    ds.write_dataset(data=table_orig, base_dir=original_dir_path, format=pq_write_format, file_options=write_options)
    pq_scan_opts_crc = ds.ParquetFragmentScanOptions(page_checksum_verification=True)
    pq_read_format_crc = pa.dataset.ParquetFileFormat(default_fragment_scan_options=pq_scan_opts_crc)
    table_check = ds.dataset(original_dir_path, format=pq_read_format_crc).to_table()
    assert table_orig == table_check
    corrupted_dir_path = tempdir / 'corrupted_dir'
    copytree(original_dir_path, corrupted_dir_path)
    corrupted_file_path_list = list(corrupted_dir_path.iterdir())
    assert len(corrupted_file_path_list) == 1
    corrupted_file_path = corrupted_file_path_list[0]
    bin_data = bytearray(corrupted_file_path.read_bytes())
    assert bin_data[31] != bin_data[36]
    bin_data[31], bin_data[36] = (bin_data[36], bin_data[31])
    corrupted_file_path.write_bytes(bin_data)
    pq_scan_opts_no_crc = ds.ParquetFragmentScanOptions(page_checksum_verification=False)
    pq_read_format_no_crc = pa.dataset.ParquetFileFormat(default_fragment_scan_options=pq_scan_opts_no_crc)
    table_corrupt = ds.dataset(corrupted_dir_path, format=pq_read_format_no_crc).to_table()
    assert table_corrupt != table_orig
    assert table_corrupt == pa.table({'a': [1, 3, 2, 4]})
    with pytest.raises(OSError, match='CRC checksum verification'):
        _ = ds.dataset(corrupted_dir_path, format=pq_read_format_crc).to_table()