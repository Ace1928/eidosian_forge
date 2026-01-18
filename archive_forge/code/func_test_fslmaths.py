import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_fslmaths(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    math = fsl.ImageMaths()
    assert math.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        math.run()
    math.inputs.in_file = filelist[0]
    math.inputs.op_string = '-add 2.5 -mul input_volume2'
    math.inputs.out_file = 'foo_math.nii'
    assert math.cmdline == 'fslmaths %s -add 2.5 -mul input_volume2 foo_math.nii' % filelist[0]
    math2 = fsl.ImageMaths(in_file=filelist[0], op_string='-add 2.5', out_file='foo2_math.nii')
    assert math2.cmdline == 'fslmaths %s -add 2.5 foo2_math.nii' % filelist[0]