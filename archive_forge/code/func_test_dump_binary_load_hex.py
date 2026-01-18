import binascii
import math
import struct
import sys
import pytest
from shapely import wkt
from shapely.geometry import Point
from shapely.geos import geos_version
from shapely.tests.legacy.conftest import shapely20_todo
from shapely.wkb import dump, dumps, load, loads
def test_dump_binary_load_hex(some_point, tmpdir):
    """Asserts that reading a text file (hex mode) as binary fails."""
    file = tmpdir.join('test.wkb')
    with open(file, 'wb') as file_pointer:
        dump(some_point, file_pointer)
    if sys.platform == 'win32':
        with open(file, 'r') as file_pointer:
            restored = load(file_pointer, hex=True)
        assert some_point != restored
        return
    with pytest.raises((UnicodeEncodeError, UnicodeDecodeError)):
        with open(file, 'r') as file_pointer:
            load(file_pointer, hex=True)