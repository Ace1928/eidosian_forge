from copy import deepcopy
import os
from nibabel import Nifti1Image
import numpy as np
import pytest
import numpy.testing as npt
from nipype.testing import example_data
from nipype.interfaces.base import Bunch, TraitError
from nipype.algorithms.modelgen import (
def test_modelgen_sparse(tmpdir):
    filename1 = tmpdir.join('test1.nii').strpath
    filename2 = tmpdir.join('test2.nii').strpath
    Nifti1Image(np.random.rand(10, 10, 10, 50), np.eye(4)).to_filename(filename1)
    Nifti1Image(np.random.rand(10, 10, 10, 50), np.eye(4)).to_filename(filename2)
    s = SpecifySparseModel()
    s.inputs.input_units = 'secs'
    s.inputs.functional_runs = [filename1, filename2]
    s.inputs.time_repetition = 6
    info = [Bunch(conditions=['cond1'], onsets=[[0, 50, 100, 180]], durations=[[2]]), Bunch(conditions=['cond1'], onsets=[[30, 40, 100, 150]], durations=[[1]])]
    s.inputs.subject_info = info
    s.inputs.volumes_in_cluster = 1
    s.inputs.time_acquisition = 2
    s.inputs.high_pass_filter_cutoff = np.inf
    res = s.run()
    assert len(res.outputs.session_info) == 2
    assert len(res.outputs.session_info[0]['regress']) == 1
    assert len(res.outputs.session_info[0]['cond']) == 0
    s.inputs.stimuli_as_impulses = False
    res = s.run()
    assert res.outputs.session_info[0]['regress'][0]['val'][0] == 1.0
    s.inputs.model_hrf = True
    res = s.run()
    npt.assert_almost_equal(res.outputs.session_info[0]['regress'][0]['val'][0], 0.016675298129743384)
    assert len(res.outputs.session_info[0]['regress']) == 1
    s.inputs.use_temporal_deriv = True
    res = s.run()
    assert len(res.outputs.session_info[0]['regress']) == 2
    npt.assert_almost_equal(res.outputs.session_info[0]['regress'][0]['val'][0], 0.016675298129743384)
    npt.assert_almost_equal(res.outputs.session_info[1]['regress'][1]['val'][5], 0.007671459162258378)