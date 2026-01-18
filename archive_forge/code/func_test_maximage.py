import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_maximage(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    maxer = fsl.MaxImage(in_file='a.nii', out_file='b.nii')
    assert maxer.cmd == 'fslmaths'
    assert maxer.cmdline == 'fslmaths a.nii -Tmax b.nii'
    cmdline = 'fslmaths a.nii -{}max b.nii'
    for dim in ['X', 'Y', 'Z', 'T']:
        maxer.inputs.dimension = dim
        assert maxer.cmdline == cmdline.format(dim)
    maxer = fsl.MaxImage(in_file='a.nii')
    assert maxer.cmdline == 'fslmaths a.nii -Tmax {}'.format(os.path.join(testdir, 'a_max{}'.format(out_ext)))