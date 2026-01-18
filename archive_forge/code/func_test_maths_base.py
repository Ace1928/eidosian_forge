import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_maths_base(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    maths = fsl.MathsCommand()
    assert maths.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        maths.run()
    maths.inputs.in_file = 'a.nii'
    out_file = 'a_maths{}'.format(out_ext)
    assert maths.cmdline == 'fslmaths a.nii {}'.format(os.path.join(testdir, out_file))
    dtypes = ['float', 'char', 'int', 'short', 'double', 'input']
    int_cmdline = 'fslmaths -dt {} a.nii ' + os.path.join(testdir, out_file)
    out_cmdline = 'fslmaths a.nii ' + os.path.join(testdir, out_file) + ' -odt {}'
    duo_cmdline = 'fslmaths -dt {} a.nii ' + os.path.join(testdir, out_file) + ' -odt {}'
    for dtype in dtypes:
        foo = fsl.MathsCommand(in_file='a.nii', internal_datatype=dtype)
        assert foo.cmdline == int_cmdline.format(dtype)
        bar = fsl.MathsCommand(in_file='a.nii', output_datatype=dtype)
        assert bar.cmdline == out_cmdline.format(dtype)
        foobar = fsl.MathsCommand(in_file='a.nii', internal_datatype=dtype, output_datatype=dtype)
        assert foobar.cmdline == duo_cmdline.format(dtype, dtype)
    maths.inputs.out_file = 'b.nii'
    assert maths.cmdline == 'fslmaths a.nii b.nii'