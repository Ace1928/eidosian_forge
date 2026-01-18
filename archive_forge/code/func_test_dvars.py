import os
import pytest
from nipype.testing import example_data
from nipype.algorithms.confounds import FramewiseDisplacement, ComputeDVARS, is_outlier
import numpy as np
@pytest.mark.skipif(nonitime, reason='nitime is not installed')
def test_dvars(tmpdir):
    ground_truth = np.loadtxt(example_data('ds003_sub-01_mc.DVARS'))
    dvars = ComputeDVARS(in_file=example_data('ds003_sub-01_mc.nii.gz'), in_mask=example_data('ds003_sub-01_mc_brainmask.nii.gz'), save_all=True, intensity_normalization=0)
    tmpdir.chdir()
    res = dvars.run()
    dv1 = np.loadtxt(res.outputs.out_all, skiprows=1)
    assert np.abs(dv1[:, 0] - ground_truth[:, 0]).sum() / len(dv1) < 0.05
    assert np.abs(dv1[:, 1] - ground_truth[:, 1]).sum() / len(dv1) < 0.05
    assert np.abs(dv1[:, 2] - ground_truth[:, 2]).sum() / len(dv1) < 0.05
    dvars = ComputeDVARS(in_file=example_data('ds003_sub-01_mc.nii.gz'), in_mask=example_data('ds003_sub-01_mc_brainmask.nii.gz'), save_all=True)
    res = dvars.run()
    dv1 = np.loadtxt(res.outputs.out_all, skiprows=1)
    assert np.abs(dv1[:, 0] - ground_truth[:, 0]).sum() / len(dv1) < 0.05
    assert np.abs(dv1[:, 1] - ground_truth[:, 1]).sum() / len(dv1) > 0.05
    assert np.abs(dv1[:, 2] - ground_truth[:, 2]).sum() / len(dv1) < 0.05