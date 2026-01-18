from nipype.interfaces.ants import (
import os
import pytest
def test_WarpTimeSeriesImageMultiTransform(change_dir, create_wtsimt):
    wtsimt = create_wtsimt
    assert wtsimt.cmdline == 'WarpTimeSeriesImageMultiTransform 4 resting.nii resting_wtsimt.nii -R ants_deformed.nii.gz ants_Warp.nii.gz ants_Affine.txt'