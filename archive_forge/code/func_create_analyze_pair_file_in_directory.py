import os
import pytest
import numpy as np
import nibabel as nb
from nipype.utils.filemanip import ensure_list
from nipype.interfaces.fsl import Info
from nipype.interfaces.fsl.base import FSLCommand
@pytest.fixture()
def create_analyze_pair_file_in_directory(request, tmpdir):
    cwd = tmpdir.chdir()
    filelist = ['a.hdr']
    analyze_pair_image_files(tmpdir.strpath, filelist, shape=(3, 3, 3, 4))

    def change_directory():
        cwd.chdir()
    request.addfinalizer(change_directory)
    return (filelist, tmpdir.strpath)