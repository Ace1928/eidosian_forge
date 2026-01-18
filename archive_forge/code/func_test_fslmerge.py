import os
import numpy as np
import nibabel as nb
import pytest
import nipype.interfaces.fsl.utils as fsl
from nipype.interfaces.fsl import no_fsl, Info
from nipype.testing.fixtures import create_files_in_directory_plus_output_type
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_fslmerge(create_files_in_directory_plus_output_type):
    filelist, outdir, _ = create_files_in_directory_plus_output_type
    merger = fsl.Merge()
    assert merger.cmd == 'fslmerge'
    with pytest.raises(ValueError):
        merger.run()
    merger.inputs.in_files = filelist
    merger.inputs.merged_file = 'foo_merged.nii'
    merger.inputs.dimension = 't'
    merger.inputs.output_type = 'NIFTI'
    assert merger.cmdline == 'fslmerge -t foo_merged.nii %s' % ' '.join(filelist)
    merger.inputs.tr = 2.25
    assert merger.cmdline == 'fslmerge -tr foo_merged.nii %s %.2f' % (' '.join(filelist), 2.25)
    merger2 = fsl.Merge(in_files=filelist, merged_file='foo_merged.nii', dimension='t', output_type='NIFTI', tr=2.25)
    assert merger2.cmdline == 'fslmerge -tr foo_merged.nii %s %.2f' % (' '.join(filelist), 2.25)