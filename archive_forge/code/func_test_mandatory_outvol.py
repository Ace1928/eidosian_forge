import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
from nipype.interfaces import freesurfer
from nipype.interfaces.freesurfer import Info
from nipype import LooseVersion
@pytest.mark.skipif(freesurfer.no_freesurfer(), reason='freesurfer is not installed')
def test_mandatory_outvol(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    mni = freesurfer.MNIBiasCorrection()
    assert mni.cmd == 'mri_nu_correct.mni'
    with pytest.raises(ValueError):
        mni.cmdline
    mni.inputs.in_file = filelist[0]
    base, ext = os.path.splitext(os.path.basename(filelist[0]))
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    assert mni.cmdline == 'mri_nu_correct.mni --i %s --n 4 --o %s_output%s' % (filelist[0], base, ext)
    mni.inputs.out_file = 'new_corrected_file.mgz'
    assert mni.cmdline == 'mri_nu_correct.mni --i %s --n 4 --o new_corrected_file.mgz' % filelist[0]
    mni2 = freesurfer.MNIBiasCorrection(in_file=filelist[0], out_file='bias_corrected_output', iterations=2)
    assert mni2.cmdline == 'mri_nu_correct.mni --i %s --n 2 --o bias_corrected_output' % filelist[0]