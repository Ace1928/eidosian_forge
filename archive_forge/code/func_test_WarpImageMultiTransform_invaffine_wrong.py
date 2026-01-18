from nipype.interfaces.ants import (
import os
import pytest
def test_WarpImageMultiTransform_invaffine_wrong(change_dir, create_wimt):
    wimt = create_wimt
    wimt.inputs.invert_affine = [3]
    with pytest.raises(Exception):
        assert wimt.cmdline