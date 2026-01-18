import os
import pytest
from nipype.testing import example_data
from nipype.algorithms.confounds import FramewiseDisplacement, ComputeDVARS, is_outlier
import numpy as np
def test_fd(tmpdir):
    tempdir = tmpdir.strpath
    ground_truth = np.loadtxt(example_data('fsl_motion_outliers_fd.txt'))
    fdisplacement = FramewiseDisplacement(in_file=example_data('fsl_mcflirt_movpar.txt'), out_file=tempdir + '/fd.txt', parameter_source='FSL')
    res = fdisplacement.run()
    with open(res.outputs.out_file) as all_lines:
        for line in all_lines:
            assert 'FramewiseDisplacement' in line
            break
    assert np.allclose(ground_truth, np.loadtxt(res.outputs.out_file, skiprows=1), atol=0.16)
    assert np.abs(ground_truth.mean() - res.outputs.fd_average) < 0.01