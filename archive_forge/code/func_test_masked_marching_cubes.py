import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
def test_masked_marching_cubes():
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.ones_like(ellipsoid_scalar, dtype=bool)
    mask[:10, :, :] = False
    mask[:, :, 20:] = False
    ver, faces, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    area = mesh_surface_area(ver, faces)
    assert_allclose(area, 299.56878662109375, rtol=0.01)