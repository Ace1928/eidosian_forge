import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
def test_marching_cubes_isotropic():
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)
    verts, faces = marching_cubes(ellipsoid_isotropic, 0.0, method='lorensen')[:2]
    surf_calc = mesh_surface_area(verts, faces)
    assert surf > surf_calc and surf_calc > surf * 0.99
    verts, faces = marching_cubes(ellipsoid_isotropic, 0.0)[:2]
    surf_calc = mesh_surface_area(verts, faces)
    assert surf > surf_calc and surf_calc > surf * 0.99