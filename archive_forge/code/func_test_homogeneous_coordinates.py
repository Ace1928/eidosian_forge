from math import prod
from pathlib import Path
from unittest import skipUnless
import numpy as np
import pytest
from nibabel import pointset as ps
from nibabel.affines import apply_affine
from nibabel.arrayproxy import ArrayProxy
from nibabel.fileslice import strided_scalar
from nibabel.onetime import auto_attr
from nibabel.optpkg import optional_package
from nibabel.spatialimages import SpatialImage
from nibabel.tests.nibabel_data import get_nibabel_data
def test_homogeneous_coordinates(self):
    ccoords = self.rng.random((5, 3))
    hcoords = np.column_stack([ccoords, np.ones(5)])
    cartesian = ps.Pointset(ccoords)
    homogeneous = ps.Pointset(hcoords, homogeneous=True)
    for points in (cartesian, homogeneous):
        assert np.array_equal(points.get_coords(), ccoords)
        assert np.array_equal(points.get_coords(as_homogeneous=True), hcoords)
    affine = np.diag([2, 3, 4, 1])
    cart2 = affine @ cartesian
    homo2 = affine @ homogeneous
    exp_c = apply_affine(affine, ccoords)
    exp_h = (affine @ hcoords.T).T
    for points in (cart2, homo2):
        assert np.array_equal(points.get_coords(), exp_c)
        assert np.array_equal(points.get_coords(as_homogeneous=True), exp_h)