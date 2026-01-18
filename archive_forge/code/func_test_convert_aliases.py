import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline import convert
from nibabel.testing import get_test_data
@pytest.mark.parametrize('orig_dtype,alias,expected_dtype', [('int64', 'mask', 'uint8'), ('int64', 'compat', 'int32'), ('int64', 'smallest', 'uint8'), ('float64', 'mask', 'uint8'), ('float64', 'compat', 'float32')])
def test_convert_aliases(tmp_path, orig_dtype, alias, expected_dtype):
    orig_fname = tmp_path / 'orig.nii'
    out_fname = tmp_path / 'out.nii'
    arr = np.arange(24).reshape((2, 3, 4))
    img = nib.Nifti1Image(arr, np.eye(4), dtype=orig_dtype)
    img.to_filename(orig_fname)
    assert orig_fname.exists()
    assert not out_fname.exists()
    convert.main([str(orig_fname), str(out_fname), '--out-dtype', alias])
    assert out_fname.is_file()
    expected_dtype = np.dtype(expected_dtype).newbyteorder(img.header.endianness)
    converted = nib.load(out_fname)
    assert converted.get_data_dtype() == expected_dtype