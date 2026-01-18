from math import pi
import pytest
from shapely.geometry import Point
from shapely.wkt import dump, dumps, load, loads
def test_dump_load_null_geometry(empty_geometry, tmpdir):
    file = tmpdir.join('test.wkt')
    with open(file, 'w') as file_pointer:
        dump(empty_geometry, file_pointer)
    with open(file, 'r') as file_pointer:
        restored = load(file_pointer)
    assert empty_geometry.equals(restored)