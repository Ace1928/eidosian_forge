import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
from nipype.interfaces import freesurfer
from nipype.interfaces.freesurfer import Info
from nipype import LooseVersion
@pytest.mark.skipif(freesurfer.no_freesurfer(), reason='freesurfer is not installed')
def test_bbregister(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    bbr = freesurfer.BBRegister()
    assert bbr.cmd == 'bbregister'
    with pytest.raises(ValueError):
        bbr.cmdline
    bbr.inputs.subject_id = 'fsaverage'
    bbr.inputs.source_file = filelist[0]
    bbr.inputs.contrast_type = 't2'
    if Info.looseversion() < LooseVersion('6.0.0'):
        with pytest.raises(ValueError):
            bbr.cmdline
    else:
        bbr.cmdline
    bbr.inputs.init = 'fsl'
    base, ext = os.path.splitext(os.path.basename(filelist[0]))
    if ext == '.gz':
        base, _ = os.path.splitext(base)
    assert bbr.cmdline == 'bbregister --t2 --init-fsl --reg {base}_bbreg_fsaverage.dat --mov {full} --s fsaverage'.format(full=filelist[0], base=base)