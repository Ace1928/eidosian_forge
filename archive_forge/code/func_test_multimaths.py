import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_multimaths(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    maths = fsl.MultiImageMaths(in_file='a.nii', out_file='c.nii')
    assert maths.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        maths.run()
    maths.inputs.operand_files = ['a.nii', 'b.nii']
    opstrings = ['-add %s -div %s', '-max 1 -sub %s -min %s', '-mas %s -add %s']
    for ostr in opstrings:
        maths.inputs.op_string = ostr
        assert maths.cmdline == 'fslmaths a.nii %s c.nii' % ostr % ('a.nii', 'b.nii')
    maths = fsl.MultiImageMaths(in_file='a.nii', op_string='-add %s -mul 5', operand_files=['b.nii'])
    assert maths.cmdline == 'fslmaths a.nii -add b.nii -mul 5 %s' % os.path.join(testdir, 'a_maths%s' % out_ext)