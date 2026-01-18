import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline import convert
from nibabel.testing import get_test_data
@pytest.mark.parametrize('ext,img_class', [('mgh', nib.MGHImage), ('img', nib.Nifti1Pair)])
def test_convert_by_extension(tmp_path, ext, img_class):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / f'output.{ext}'
    orig = nib.load(infile)
    assert not outfile.exists()
    convert.main([str(infile), str(outfile)])
    assert outfile.is_file()
    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    assert converted.__class__ == img_class