import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
from cartopy import geodesic
def test_geometry_length_ndarray():
    geod = geodesic.Geodesic()
    geom = np.array([lhr, jfk, lhr])
    expected = pytest.approx(lhr_to_jfk * 2, abs=1)
    assert geod.geometry_length(geom) == expected