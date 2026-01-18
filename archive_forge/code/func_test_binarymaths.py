import os
import numpy as np
from nipype.interfaces.base import Undefined
import nipype.interfaces.fsl.maths as fsl
from nipype.interfaces.fsl import no_fsl
import pytest
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_binarymaths(create_files_in_directory_plus_output_type):
    files, testdir, out_ext = create_files_in_directory_plus_output_type
    maths = fsl.BinaryMaths(in_file='a.nii', out_file='c.nii')
    assert maths.cmd == 'fslmaths'
    with pytest.raises(ValueError):
        maths.run()
    ops = ['add', 'sub', 'mul', 'div', 'rem', 'min', 'max']
    operands = ['b.nii', -2, -0.5, 0, 0.123456, np.pi, 500]
    for op in ops:
        for ent in operands:
            maths = fsl.BinaryMaths(in_file='a.nii', out_file='c.nii', operation=op)
            if ent == 'b.nii':
                maths.inputs.operand_file = ent
                assert maths.cmdline == 'fslmaths a.nii -{} b.nii c.nii'.format(op)
            else:
                maths.inputs.operand_value = ent
                assert maths.cmdline == 'fslmaths a.nii -{} {:.8f} c.nii'.format(op, ent)
    for op in ops:
        maths = fsl.BinaryMaths(in_file='a.nii', operation=op, operand_file='b.nii')
        assert maths.cmdline == 'fslmaths a.nii -{} b.nii {}'.format(op, os.path.join(testdir, 'a_maths{}'.format(out_ext)))