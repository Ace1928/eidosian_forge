import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
def test_masked_marching_cubes_all_true():
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.ones_like(ellipsoid_scalar, dtype=bool)
    ver_m, faces_m, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    ver, faces, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    assert_allclose(ver_m, ver, rtol=1e-05)
    assert_allclose(faces_m, faces, rtol=1e-05)