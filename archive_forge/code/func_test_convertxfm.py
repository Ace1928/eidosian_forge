import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_convertxfm(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    cvt = fsl.ConvertXFM()
    assert cvt.cmd == 'convert_xfm'
    with pytest.raises(ValueError):
        cvt.run()
    cvt.inputs.in_file = filelist[0]
    cvt.inputs.invert_xfm = True
    cvt.inputs.out_file = 'foo.mat'
    assert cvt.cmdline == 'convert_xfm -omat foo.mat -inverse %s' % filelist[0]
    cvt2 = fsl.ConvertXFM(in_file=filelist[0], in_file2=filelist[1], concat_xfm=True, out_file='bar.mat')
    assert cvt2.cmdline == 'convert_xfm -omat bar.mat -concat %s %s' % (filelist[1], filelist[0])