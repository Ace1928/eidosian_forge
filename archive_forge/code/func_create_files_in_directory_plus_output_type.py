import os
import pytest
import numpy as np
import nibabel as nb
from nipype.utils.filemanip import ensure_list
from nipype.interfaces.fsl import Info
from nipype.interfaces.fsl.base import FSLCommand
@pytest.fixture(params=[None] + sorted(Info.ftypes))
def create_files_in_directory_plus_output_type(request, tmpdir):
    func_prev_type = set_output_type(request.param)
    origdir = tmpdir.chdir()
    filelist = ['a.nii', 'b.nii']
    nifti_image_files(tmpdir.strpath, filelist, shape=(3, 3, 3, 4))
    out_ext = Info.output_type_to_ext(Info.output_type())

    def fin():
        set_output_type(func_prev_type)
        origdir.chdir()
    request.addfinalizer(fin)
    return (filelist, tmpdir.strpath, out_ext)