from nipype.interfaces.ants import (
import os
import pytest
def test_WarpTimeSeriesImageMultiTransform_invaffine(change_dir, create_wtsimt):
    wtsimt = create_wtsimt
    wtsimt.inputs.invert_affine = [1]
    assert wtsimt.cmdline == 'WarpTimeSeriesImageMultiTransform 4 resting.nii resting_wtsimt.nii -R ants_deformed.nii.gz ants_Warp.nii.gz -i ants_Affine.txt'