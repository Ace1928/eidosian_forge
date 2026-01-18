from nipype.interfaces.ants import (
import os
import pytest
@pytest.fixture()
def create_wimt():
    wimt = WarpImageMultiTransform()
    wimt.inputs.input_image = 'diffusion_weighted.nii'
    wimt.inputs.reference_image = 'functional.nii'
    wimt.inputs.transformation_series = ['func2anat_coreg_Affine.txt', 'func2anat_InverseWarp.nii.gz', 'dwi2anat_Warp.nii.gz', 'dwi2anat_coreg_Affine.txt']
    return wimt