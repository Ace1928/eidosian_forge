import os
import nipype.interfaces.fsl as fsl
from nipype.interfaces.base import InterfaceResult
from nipype.interfaces.fsl import check_fsl, no_fsl
import pytest
@pytest.mark.skipif(no_fsl(), reason='fsl is not installed')
def test_FSLCommand():
    cmd = fsl.FSLCommand(command='ls')
    res = cmd.run()
    assert type(res) == InterfaceResult