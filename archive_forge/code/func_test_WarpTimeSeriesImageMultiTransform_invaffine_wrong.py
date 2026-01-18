from nipype.interfaces.ants import (
import os
import pytest
def test_WarpTimeSeriesImageMultiTransform_invaffine_wrong(change_dir, create_wtsimt):
    wtsimt = create_wtsimt
    wtsimt.inputs.invert_affine = [0]
    with pytest.raises(Exception):
        wtsimt.cmdline