from copy import deepcopy
import os
from nibabel import Nifti1Image
import numpy as np
import pytest
import numpy.testing as npt
from nipype.testing import example_data
from nipype.interfaces.base import Bunch, TraitError
from nipype.algorithms.modelgen import (
def test_modelgen_spm_concat(tmpdir):
    filename1 = tmpdir.join('test1.nii').strpath
    filename2 = tmpdir.join('test2.nii').strpath
    Nifti1Image(np.random.rand(10, 10, 10, 30), np.eye(4)).to_filename(filename1)
    Nifti1Image(np.random.rand(10, 10, 10, 30), np.eye(4)).to_filename(filename2)
    s = SpecifySPMModel()
    s.inputs.input_units = 'secs'
    s.inputs.concatenate_runs = True
    setattr(s.inputs, 'output_units', 'secs')
    assert s.inputs.output_units == 'secs'
    s.inputs.functional_runs = [filename1, filename2]
    s.inputs.time_repetition = 6
    s.inputs.high_pass_filter_cutoff = 128.0
    info = [Bunch(conditions=['cond1'], onsets=[[2, 50, 100, 170]], durations=[[1]]), Bunch(conditions=['cond1'], onsets=[[30, 40, 100, 150]], durations=[[1]])]
    s.inputs.subject_info = deepcopy(info)
    res = s.run()
    assert len(res.outputs.session_info) == 1
    assert len(res.outputs.session_info[0]['regress']) == 1
    assert np.sum(res.outputs.session_info[0]['regress'][0]['val']) == 30
    assert len(res.outputs.session_info[0]['cond']) == 1
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][0]['onset']), np.array([2.0, 50.0, 100.0, 170.0, 210.0, 220.0, 280.0, 330.0]))
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][0]['duration']), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    setattr(s.inputs, 'output_units', 'scans')
    assert s.inputs.output_units == 'scans'
    s.inputs.subject_info = deepcopy(info)
    res = s.run()
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][0]['onset']), np.array([2.0, 50.0, 100.0, 170.0, 210.0, 220.0, 280.0, 330.0]) / 6)
    s.inputs.concatenate_runs = False
    s.inputs.subject_info = deepcopy(info)
    s.inputs.output_units = 'secs'
    res = s.run()
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][0]['onset']), np.array([2.0, 50.0, 100.0, 170.0]))
    filename3 = tmpdir.join('test3.nii').strpath
    Nifti1Image(np.random.rand(10, 10, 10, 30), np.eye(4)).to_filename(filename3)
    s.inputs.functional_runs = [filename1, filename2, filename3]
    info = [Bunch(conditions=['cond1', 'cond2'], onsets=[[2, 3], [2]], durations=[[1, 1], [1]]), Bunch(conditions=['cond1', 'cond2'], onsets=[[2, 3], [2, 4]], durations=[[1, 1], [1, 1]]), Bunch(conditions=['cond1', 'cond2'], onsets=[[2, 3], [2]], durations=[[1, 1], [1]])]
    s.inputs.subject_info = deepcopy(info)
    res = s.run()
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][0]['duration']), np.array([1.0, 1.0]))
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][1]['duration']), np.array([1.0]))
    npt.assert_almost_equal(np.array(res.outputs.session_info[1]['cond'][1]['duration']), np.array([1.0, 1.0]))
    npt.assert_almost_equal(np.array(res.outputs.session_info[2]['cond'][1]['duration']), np.array([1.0]))
    s.inputs.concatenate_runs = True
    info = [Bunch(conditions=['cond1', 'cond2'], onsets=[[2, 3], [2]], durations=[[1, 1], [1]]), Bunch(conditions=['cond1', 'cond2'], onsets=[[2, 3], [2, 4]], durations=[[1, 1], [1, 1]]), Bunch(conditions=['cond1', 'cond2'], onsets=[[2, 3], [2]], durations=[[1, 1], [1]])]
    s.inputs.subject_info = deepcopy(info)
    res = s.run()
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][0]['duration']), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    npt.assert_almost_equal(np.array(res.outputs.session_info[0]['cond'][1]['duration']), np.array([1.0, 1.0, 1.0, 1.0]))