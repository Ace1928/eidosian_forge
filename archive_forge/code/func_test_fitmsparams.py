import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
from nipype.interfaces import freesurfer
from nipype.interfaces.freesurfer import Info
from nipype import LooseVersion
@pytest.mark.skipif(freesurfer.no_freesurfer(), reason='freesurfer is not installed')
def test_fitmsparams(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    fit = freesurfer.FitMSParams()
    assert fit.cmd == 'mri_ms_fitparms'
    with pytest.raises(ValueError):
        fit.run()
    fit.inputs.in_files = filelist
    fit.inputs.out_dir = outdir
    assert fit.cmdline == 'mri_ms_fitparms  %s %s %s' % (filelist[0], filelist[1], outdir)
    fit2 = freesurfer.FitMSParams(in_files=filelist, te_list=[1.5, 3.5], flip_list=[20, 30], out_dir=outdir)
    assert fit2.cmdline == 'mri_ms_fitparms  -te %.3f -fa %.1f %s -te %.3f -fa %.1f %s %s' % (1.5, 20.0, filelist[0], 3.5, 30.0, filelist[1], outdir)