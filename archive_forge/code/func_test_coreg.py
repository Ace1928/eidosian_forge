import os
import pytest
from nipype.testing import example_data
import nipype.interfaces.spm.utils as spmu
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import split_filename, fname_presuffix
from nipype.interfaces.base import TraitError
def test_coreg():
    moving = example_data(infile='functional.nii')
    target = example_data(infile='T1.nii')
    mat = example_data(infile='trans.mat')
    coreg = spmu.CalcCoregAffine(matlab_cmd='mymatlab')
    coreg.inputs.target = target
    assert coreg.inputs.matlab_cmd == 'mymatlab'
    coreg.inputs.moving = moving
    assert not isdefined(coreg.inputs.mat)
    pth, mov, _ = split_filename(moving)
    _, tgt, _ = split_filename(target)
    mat = os.path.join(pth, '%s_to_%s.mat' % (mov, tgt))
    invmat = fname_presuffix(mat, prefix='inverse_')
    scrpt = coreg._make_matlab_command(None)
    assert coreg.inputs.mat == mat
    assert coreg.inputs.invmat == invmat