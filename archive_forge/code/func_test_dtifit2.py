import os
import nipype.interfaces.fsl.dti as fsl
from nipype.interfaces.fsl import Info, no_fsl
from nipype.interfaces.base import Undefined
import pytest
from nipype.testing.fixtures import create_files_in_directory
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_dtifit2(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    dti = fsl.DTIFit()
    assert dti.cmd == 'dtifit'
    with pytest.raises(ValueError):
        dti.run()
    dti.inputs.dwi = filelist[0]
    dti.inputs.base_name = 'foo.dti.nii'
    dti.inputs.mask = filelist[1]
    dti.inputs.bvecs = filelist[0]
    dti.inputs.bvals = filelist[1]
    dti.inputs.min_z = 10
    dti.inputs.max_z = 50
    assert dti.cmdline == 'dtifit -k %s -o foo.dti.nii -m %s -r %s -b %s -Z 50 -z 10' % (filelist[0], filelist[1], filelist[0], filelist[1])