import numpy as np
import nibabel as nb
from nipype.algorithms.stats import ActivationCount
import pytest
def test_ActivationCount(tmpdir):
    tmpdir.chdir()
    in_files = ['{:d}.nii'.format(i) for i in range(3)]
    for fname in in_files:
        nb.Nifti1Image(np.random.normal(size=(5, 5, 5)), np.eye(4)).to_filename(fname)
    acm = ActivationCount(in_files=in_files, threshold=1.65)
    res = acm.run()
    diff = nb.load(res.outputs.out_file)
    pos = nb.load(res.outputs.acm_pos)
    neg = nb.load(res.outputs.acm_neg)
    assert np.allclose(diff.get_fdata(), pos.get_fdata() - neg.get_fdata())