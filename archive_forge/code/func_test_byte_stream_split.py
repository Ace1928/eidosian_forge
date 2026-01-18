from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
def test_byte_stream_split():
    arr_float = pa.array(list(map(float, range(100))))
    arr_int = pa.array(list(map(int, range(100))))
    data_float = [arr_float, arr_float]
    table = pa.Table.from_arrays(data_float, names=['a', 'b'])
    _check_roundtrip(table, expected=table, compression='gzip', use_dictionary=False, use_byte_stream_split=True)
    _check_roundtrip(table, expected=table, compression='gzip', use_dictionary=['a'], use_byte_stream_split=['b'])
    _check_roundtrip(table, expected=table, compression='gzip', use_dictionary=['a', 'b'], use_byte_stream_split=['a', 'b'])
    mixed_table = pa.Table.from_arrays([arr_float, arr_int], names=['a', 'b'])
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=['b'], use_byte_stream_split=['a'])
    table = pa.Table.from_arrays([arr_int], names=['tmp'])
    with pytest.raises(IOError):
        _check_roundtrip(table, expected=table, use_byte_stream_split=True, use_dictionary=False)