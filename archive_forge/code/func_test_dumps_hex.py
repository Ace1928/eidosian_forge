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
def test_dumps_hex(some_point):
    result = dumps(some_point, hex=True)
    assert result == hostorder('BIdd', '0101000000333333333333F33F3333333333330B40')