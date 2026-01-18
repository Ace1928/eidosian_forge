import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_affine_3d(self):
    g2 = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
    g3 = load_wkt('LINESTRING(2.4 4.1 100.2, 2.4 3 132.8, 3 3 128.6)')
    matrix2d = (2, 0, 0, 2.5, -5, 4.1)
    matrix3d = (2, 0, 0, 0, 2.5, 0, 0, 0, 0.3048, -5, 4.1, 100)
    a22 = affinity.affine_transform(g2, matrix2d)
    a23 = affinity.affine_transform(g2, matrix3d)
    a32 = affinity.affine_transform(g3, matrix2d)
    a33 = affinity.affine_transform(g3, matrix3d)
    assert not a22.has_z
    assert not a23.has_z
    assert a32.has_z
    assert a33.has_z
    expected2d = load_wkt('LINESTRING(-0.2 14.35, -0.2 11.6, 1 11.6)')
    expected3d = load_wkt('LINESTRING(-0.2 14.35 130.54096, -0.2 11.6 140.47744, 1 11.6 139.19728)')
    expected32 = load_wkt('LINESTRING(-0.2 14.35 100.2, -0.2 11.6 132.8, 1 11.6 128.6)')
    assert a22.equals_exact(expected2d, 1e-06)
    assert a23.equals_exact(expected2d, 1e-06)
    for a, e in zip(a32.coords, expected32.coords):
        for ap, ep in zip(a, e):
            self.assertAlmostEqual(ap, ep)
    for a, e in zip(a33.coords, expected3d.coords):
        for ap, ep in zip(a, e):
            self.assertAlmostEqual(ap, ep)