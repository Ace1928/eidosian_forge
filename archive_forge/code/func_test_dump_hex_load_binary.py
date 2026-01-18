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
@shapely20_todo
def test_dump_hex_load_binary(some_point, tmpdir):
    """Asserts that reading a binary file as text (hex mode) fails."""
    file = tmpdir.join('test.wkb')
    with open(file, 'w') as file_pointer:
        dump(some_point, file_pointer, hex=True)
    with pytest.raises(TypeError):
        with open(file, 'rb') as file_pointer:
            load(file_pointer)