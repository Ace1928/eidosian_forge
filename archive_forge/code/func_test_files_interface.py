from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import MGHImage, Nifti1Image, Nifti1Pair, all_image_classes
from ..fileholders import FileHolderError
from ..spatialimages import SpatialImage
def test_files_interface():
    arr = np.zeros((2, 3, 4))
    aff = np.eye(4)
    img = Nifti1Image(arr, aff)
    img.set_filename('test')
    assert img.get_filename() == 'test.nii'
    assert img.file_map['image'].filename == 'test.nii'
    with pytest.raises(KeyError):
        img.file_map['header']
    img = Nifti1Pair(arr, aff)
    img.set_filename('test')
    assert img.get_filename() == 'test.img'
    assert img.file_map['image'].filename == 'test.img'
    assert img.file_map['header'].filename == 'test.hdr'
    img = Nifti1Image(arr, aff)
    img.file_map['image'].fileobj = BytesIO()
    img.to_file_map()
    img2 = Nifti1Image.from_file_map(img.file_map)
    assert_array_equal(img2.get_fdata(), img.get_fdata())
    img = Nifti1Pair(arr, aff)
    img.file_map['image'].fileobj = BytesIO()
    with pytest.raises(FileHolderError):
        img.to_file_map()
    img.file_map['header'].fileobj = BytesIO()
    img.to_file_map()
    img2 = Nifti1Pair.from_file_map(img.file_map)
    assert_array_equal(img2.get_fdata(), img.get_fdata())