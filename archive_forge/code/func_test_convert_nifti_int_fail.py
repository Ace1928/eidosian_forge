import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline import convert
from nibabel.testing import get_test_data
def test_convert_nifti_int_fail(tmp_path):
    infile = get_test_data(fname='anatomical.nii')
    outfile = tmp_path / f'output.nii'
    orig = nib.load(infile)
    assert not outfile.exists()
    with pytest.raises(ValueError):
        convert.main([str(infile), str(outfile), '--out-dtype', 'int'])
    assert not outfile.exists()
    with pytest.warns(UserWarning):
        convert.main([str(infile), str(outfile), '--out-dtype', 'int', '--force'])
    assert outfile.is_file()
    converted = nib.load(outfile)
    assert np.allclose(converted.affine, orig.affine)
    assert converted.shape == orig.shape
    assert converted.get_data_dtype() == orig.get_data_dtype()