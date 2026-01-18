import numpy as np
import nibabel as nb
from nipype.algorithms.stats import ActivationCount
import pytest
@pytest.mark.parametrize('threshold, above_thresh', [(1, 15.865), (2, 2.275), (3, 0.135)])
def test_ActivationCount_normaldistr(tmpdir, threshold, above_thresh):
    tmpdir.chdir()
    in_files = ['{:d}.nii'.format(i) for i in range(3)]
    for fname in in_files:
        nb.Nifti1Image(np.random.normal(size=(100, 100, 100)), np.eye(4)).to_filename(fname)
    acm = ActivationCount(in_files=in_files, threshold=threshold)
    res = acm.run()
    pos = nb.load(res.outputs.acm_pos)
    neg = nb.load(res.outputs.acm_neg)
    assert np.isclose(pos.get_fdata().mean(), above_thresh * 0.01, rtol=0.1, atol=0.0001)
    assert np.isclose(neg.get_fdata().mean(), above_thresh * 0.01, rtol=0.1, atol=0.0001)