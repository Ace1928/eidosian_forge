import os
import nipype.interfaces.fsl as fsl
from nipype.interfaces.base import InterfaceResult
from nipype.interfaces.fsl import check_fsl, no_fsl
import pytest
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
@pytest.mark.parametrize('args, desired_name', [({}, {'file': 'foo.nii.gz'}), ({'suffix': '_brain'}, {'file': 'foo_brain.nii.gz'}), ({'suffix': '_brain', 'cwd': '/data'}, {'dir': '/data', 'file': 'foo_brain.nii.gz'}), ({'suffix': '_brain.mat', 'change_ext': False}, {'file': 'foo_brain.mat'})])
def test_gen_fname(args, desired_name):
    cmd = fsl.FSLCommand(command='junk', output_type='NIFTI_GZ')
    pth = os.getcwd()
    fname = cmd._gen_fname('foo.nii.gz', **args)
    if 'dir' in desired_name.keys():
        desired = os.path.join(desired_name['dir'], desired_name['file'])
    else:
        desired = os.path.join(pth, desired_name['file'])
    assert fname == desired