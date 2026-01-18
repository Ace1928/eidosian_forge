import os
import pytest
import numpy as np
import nibabel as nb
from nipype.utils.filemanip import ensure_list
from nipype.interfaces.fsl import Info
from nipype.interfaces.fsl.base import FSLCommand
@pytest.fixture()
def create_files_in_directory_plus_dummy_file(request, tmpdir):
    cwd = tmpdir.chdir()
    filelist = ['a.nii', 'b.nii']
    nifti_image_files(tmpdir.strpath, filelist, shape=(3, 3, 3, 4))
    tmpdir.join('reg.dat').write('dummy file')
    filelist.append('reg.dat')

    def change_directory():
        cwd.chdir()
    request.addfinalizer(change_directory)
    return (filelist, tmpdir.strpath)