import os
import os.path as op
import pytest
from nipype.testing.fixtures import (
from nipype.pipeline import engine as pe
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import TraitError
from nipype.interfaces.io import FreeSurferSource
@pytest.mark.skipif(fs.no_freesurfer(), reason='freesurfer is not installed')
def test_eulernumber(tmpdir):
    fssrc = FreeSurferSource(subjects_dir=fs.Info.subjectsdir(), subject_id='fsaverage', hemi='lh')
    pial = fssrc.run().outputs.pial
    assert isinstance(pial, str), 'Problem when fetching surface file'
    eu = fs.EulerNumber()
    eu.inputs.in_file = pial
    res = eu.run()
    assert res.outputs.defects == 0
    assert res.outputs.euler == 2