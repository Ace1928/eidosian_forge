from ..segmentation import LaplacianThickness
from .test_resampling import change_dir
import os
import pytest
def test_LaplacianThickness_wrongargs(change_dir, create_lt):
    lt = create_lt
    lt.inputs.tolerance = 0.001
    with pytest.raises(ValueError, match=".* requires a value for input 'sulcus_prior' .*"):
        lt.cmdline
    lt.inputs.sulcus_prior = 0.15
    with pytest.raises(ValueError, match=".* requires a value for input 'dT' .*"):
        lt.cmdline
    lt.inputs.dT = 0.01
    with pytest.raises(ValueError, match=".* requires a value for input 'prior_thickness' .*"):
        lt.cmdline
    lt.inputs.prior_thickness = 5.9
    with pytest.raises(ValueError, match=".* requires a value for input 'smooth_param' .*"):
        lt.cmdline
    lt.inputs.smooth_param = 4.5
    assert lt.cmdline == 'LaplacianThickness functional.nii diffusion_weighted.nii functional_thickness.nii 4.5 5.9 0.01 0.15 0.001'