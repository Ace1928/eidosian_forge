import os
import pytest
from nipype.testing.fixtures import create_files_in_directory
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab
def test_coregister_list_outputs(create_files_in_directory):
    filelist, outdir = create_files_in_directory
    coreg = spm.Coregister(source=filelist[0])
    assert coreg._list_outputs()['coregistered_source'][0].startswith('r')
    coreg = spm.Coregister(source=filelist[0], apply_to_files=filelist[1])
    assert coreg._list_outputs()['coregistered_files'][0].startswith('r')