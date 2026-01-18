import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def test_dpconn():
    parcel_map = create_parcel_map((0,))
    geometry_map = create_geometry_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((parcel_map, geometry_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(4, 10)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_PARCELLATED')
    with InTemporaryDirectory():
        ci.save(img, 'test.dpconn.nii')
        img2 = ci.load('test.dpconn.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnDenseParcel'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2