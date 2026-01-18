import os
from copy import deepcopy
import pytest
import pdb
from nipype.utils.filemanip import split_filename, ensure_list
from .. import preprocess as fsl
from nipype.interfaces.fsl import Info
from nipype.interfaces.base import File, TraitError, Undefined, isdefined
from nipype.interfaces.fsl import no_fsl
@pytest.fixture()
def setup_fugue(tmpdir):
    import nibabel as nb
    import numpy as np
    import os.path as op
    d = np.ones((80, 80, 80))
    infile = tmpdir.join('dumbfile.nii.gz').strpath
    nb.Nifti1Image(d, None, None).to_filename(infile)
    return (tmpdir, infile)