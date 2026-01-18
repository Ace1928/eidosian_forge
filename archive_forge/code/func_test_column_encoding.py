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
def test_column_encoding():
    arr_float = pa.array(list(map(float, range(100))))
    arr_int = pa.array(list(map(int, range(100))))
    arr_bin = pa.array([str(x) for x in range(100)], type=pa.binary())
    arr_flba = pa.array([str(x).zfill(10) for x in range(100)], type=pa.binary(10))
    arr_bool = pa.array([False, True, False, False] * 25)
    mixed_table = pa.Table.from_arrays([arr_float, arr_int, arr_bin, arr_flba, arr_bool], names=['a', 'b', 'c', 'd', 'e'])
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'a': 'BYTE_STREAM_SPLIT', 'b': 'PLAIN', 'c': 'PLAIN'})
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding='PLAIN')
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'a': 'PLAIN', 'b': 'DELTA_BINARY_PACKED', 'c': 'PLAIN'})
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'a': 'PLAIN', 'b': 'DELTA_BINARY_PACKED', 'c': 'DELTA_LENGTH_BYTE_ARRAY'})
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'a': 'PLAIN', 'b': 'DELTA_BINARY_PACKED', 'c': 'DELTA_BYTE_ARRAY', 'd': 'DELTA_BYTE_ARRAY'})
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'e': 'RLE'})
    with pytest.raises(IOError, match='BYTE_STREAM_SPLIT only supports FLOAT and DOUBLE'):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'a': 'PLAIN', 'b': 'BYTE_STREAM_SPLIT', 'c': 'PLAIN'})
    with pytest.raises(OSError):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'a': 'DELTA_BINARY_PACKED', 'b': 'PLAIN', 'c': 'PLAIN'})
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding='RLE_DICTIONARY')
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding={'a': 'MADE_UP_ENCODING'})
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=['b'], column_encoding={'b': 'PLAIN'})
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table, column_encoding={'b': 'PLAIN'})
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, use_byte_stream_split=['a'], column_encoding={'a': 'RLE', 'b': 'BYTE_STREAM_SPLIT', 'c': 'PLAIN'})
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, use_byte_stream_split=True, column_encoding={'a': 'RLE', 'b': 'BYTE_STREAM_SPLIT', 'c': 'PLAIN'})
    with pytest.raises(TypeError):
        _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False, column_encoding=True)