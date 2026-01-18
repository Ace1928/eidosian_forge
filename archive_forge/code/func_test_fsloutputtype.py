import os
import nipype.interfaces.fsl as fsl
from nipype.interfaces.base import InterfaceResult
from nipype.interfaces.fsl import check_fsl, no_fsl
import pytest
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_fsloutputtype():
    types = list(fsl.Info.ftypes.keys())
    orig_out_type = fsl.Info.output_type()
    assert orig_out_type in types