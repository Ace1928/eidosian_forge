import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
from cartopy import geodesic
def test_geometry_length_multilinestring():
    geod = geodesic.Geodesic()
    geom = sgeom.MultiLineString([sgeom.LineString(np.array([lhr, jfk])), sgeom.LineString(np.array([tul, jfk]))])
    expected = pytest.approx(lhr_to_jfk + jfk_to_tul, abs=1)
    assert geod.geometry_length(geom) == expected