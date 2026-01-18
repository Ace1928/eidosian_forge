from nipype.interfaces.ants import (
import os
import pytest
@pytest.fixture()
def create_wtsimt():
    wtsimt = WarpTimeSeriesImageMultiTransform()
    wtsimt.inputs.input_image = 'resting.nii'
    wtsimt.inputs.reference_image = 'ants_deformed.nii.gz'
    wtsimt.inputs.transformation_series = ['ants_Warp.nii.gz', 'ants_Affine.txt']
    return wtsimt