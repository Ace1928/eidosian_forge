import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_meanimage(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    meaner = fsl.MeanImage(in_file='a.nii', out_file='b.nii')
    assert meaner.cmd == 'fslmaths'
    assert meaner.cmdline == 'fslmaths a.nii -Tmean b.nii'
    cmdline = 'fslmaths a.nii -{}mean b.nii'
    for dim in ['X', 'Y', 'Z', 'T']:
        meaner.inputs.dimension = dim
        assert meaner.cmdline == cmdline.format(dim)
    meaner = fsl.MeanImage(in_file='a.nii')
    assert meaner.cmdline == 'fslmaths a.nii -Tmean {}'.format(os.path.join(testdir, 'a_mean{}'.format(out_ext)))