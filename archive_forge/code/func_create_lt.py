from ..segmentation import LaplacianThickness
from .test_resampling import change_dir
import os
import pytest
@pytest.fixture()
def create_lt():
    lt = LaplacianThickness()
    lt.inputs.input_gm = 'diffusion_weighted.nii'
    lt.inputs.input_wm = 'functional.nii'
    return lt