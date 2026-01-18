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
def test_dumps_endianness(some_point):
    result = dumps(some_point)
    assert bin2hex(result) == hostorder('BIdd', '0101000000333333333333F33F3333333333330B40')
    result = dumps(some_point, big_endian=False)
    assert bin2hex(result) == '0101000000333333333333F33F3333333333330B40'
    result = dumps(some_point, big_endian=True)
    assert bin2hex(result) == '00000000013FF3333333333333400B333333333333'