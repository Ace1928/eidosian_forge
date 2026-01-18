import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
from cartopy import geodesic
def test_geometry_length_polygon():
    geod = geodesic.Geodesic()
    geom = sgeom.Polygon(np.array([lhr, jfk, tul]))
    expected = pytest.approx(lhr_to_jfk + jfk_to_tul + tul_to_lhr, abs=1)
    assert geod.geometry_length(geom) == expected