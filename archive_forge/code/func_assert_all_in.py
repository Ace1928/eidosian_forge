import numpy as np
import numpy.linalg as npl
import pytest
from numpy.testing import assert_almost_equal
from ..affines import apply_affine, from_matvec
from ..eulerangles import euler2mat
from ..nifti1 import Nifti1Image
from ..spaces import slice2volume, vox2out_vox
def assert_all_in(in_shape, in_affine, out_shape, out_affine):
    slices = tuple((slice(N) for N in in_shape))
    n_axes = len(in_shape)
    in_grid = np.mgrid[slices]
    in_grid = np.rollaxis(in_grid, 0, n_axes + 1)
    v2v = npl.inv(out_affine).dot(in_affine)
    if n_axes < 3:
        new_v2v = np.eye(n_axes + 1)
        new_v2v[:n_axes, :n_axes] = v2v[:n_axes, :n_axes]
        new_v2v[:n_axes, -1] = v2v[:n_axes, -1]
        v2v = new_v2v
    out_grid = apply_affine(v2v, in_grid)
    TINY = 1e-12
    assert np.all(out_grid > -TINY)
    assert np.all(out_grid < np.array(out_shape) + TINY)