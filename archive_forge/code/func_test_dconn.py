import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
def test_dconn():
    mapping = create_geometry_map((0, 1))
    matrix = ci.Cifti2Matrix()
    matrix.append(mapping)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(10, 10)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE')
    with InTemporaryDirectory():
        ci.save(img, 'test.dconn.nii')
        img2 = nib.load('test.dconn.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnDense'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        assert img2.header.matrix.get_index_map(0) == img2.header.matrix.get_index_map(1)
        check_geometry_map(img2.header.matrix.get_index_map(0))
        del img2