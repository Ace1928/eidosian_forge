import os
import unittest
from unittest import mock
import numpy as np
import pytest
import nibabel as nb
from nibabel.cmdline.roi import lossless_slice, main, parse_slice
from nibabel.testing import data_path
@pytest.mark.parametrize('inplace', (True, False))
def test_nib_roi(tmp_path, inplace):
    in_file = os.path.join(data_path, 'functional.nii')
    out_file = str(tmp_path / 'sliced.nii')
    in_img = nb.load(in_file)
    if inplace:
        in_img.to_filename(out_file)
        in_file = out_file
    retval = main([in_file, out_file, '-i', '1:-1', '-j', '-1:1:-1', '-k', '::', '-t', ':5'])
    assert retval == 0
    out_img = nb.load(out_file)
    in_data = in_img.dataobj[:]
    in_sliced = in_img.slicer[1:-1, -1:1:-1, :, :5]
    assert out_img.shape == in_sliced.shape
    assert np.array_equal(in_data[1:-1, -1:1:-1, :, :5], out_img.dataobj)
    assert np.allclose(in_sliced.dataobj, out_img.dataobj)
    assert np.allclose(in_sliced.affine, out_img.affine)