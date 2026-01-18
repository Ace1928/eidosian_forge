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
def setup_infile(tmpdir):
    ext = Info.output_type_to_ext(Info.output_type())
    tmp_infile = tmpdir.join('foo' + ext)
    tmp_infile.open('w')
    return (tmp_infile.strpath, tmpdir.strpath)