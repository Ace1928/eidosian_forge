import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..analyze import AnalyzeImage
from ..funcs import OrientationError, as_closest_canonical, concat_images
from ..loadsave import save
from ..nifti1 import Nifti1Image
from ..tmpdirs import InTemporaryDirectory
def test_closest_canonical():
    arr = np.arange(24, dtype=np.int32).reshape((2, 3, 4, 1))
    img = AnalyzeImage(arr, np.eye(4))
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img
    img = AnalyzeImage(arr, np.diag([-1, 1, 1, 1]))
    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.flipud(arr))
    img = Nifti1Image(arr, np.eye(4))
    img.header.set_dim_info(0, 1, 2)
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img
    img = Nifti1Image(arr, np.diag([-1, 1, 1, 1]))
    img.header.set_dim_info(0, 1, 2)
    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    assert img.header.get_dim_info() == xyz_img.header.get_dim_info()
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.flipud(arr))
    xyz_img = as_closest_canonical(img, True)
    aff = np.eye(4)
    aff[0, 1] = 0.1
    img = Nifti1Image(arr, aff)
    xyz_img = as_closest_canonical(img)
    assert img is xyz_img
    with pytest.raises(OrientationError):
        as_closest_canonical(img, True)
    aff = np.diag([1, 0, 0, 1])
    aff[1, 2] = 1
    aff[2, 1] = 1
    img = Nifti1Image(arr, aff)
    img.header.set_dim_info(0, 1, 2)
    xyz_img = as_closest_canonical(img)
    assert img is not xyz_img
    assert img.header.get_dim_info() == (0, 1, 2)
    assert xyz_img.header.get_dim_info() == (0, 2, 1)
    out_arr = xyz_img.get_fdata()
    assert_array_equal(out_arr, np.transpose(arr, (0, 2, 1, 3)))
    img.header.set_dim_info(None, None, 2)
    xyz_img = as_closest_canonical(img)
    assert xyz_img.header.get_dim_info() == (None, None, 1)