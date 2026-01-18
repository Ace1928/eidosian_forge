import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_affine_2d(self):
    g = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
    expected2d = load_wkt('LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)')
    matrix2d = (2, 0, 0, 2.5, -5, 4.1)
    a2 = affinity.affine_transform(g, matrix2d)
    assert a2.equals_exact(expected2d, 1e-06)
    assert not a2.has_z
    matrix3d = (2, 0, 0, 0, 2.5, 0, 0, 0, 10, -5, 4.1, 100)
    a3 = affinity.affine_transform(g, matrix3d)
    assert a3.equals_exact(expected2d, 1e-06)
    assert not a3.has_z