import os
import nipype.interfaces.fsl.dti as fsl
from nipype.interfaces.fsl import Info, no_fsl
from nipype.interfaces.base import Undefined
import pytest
from nipype.testing.fixtures import create_files_in_directory
@pytest.mark.xfail(reason='These tests are skipped until we clean up some of this code')
def test_Proj_thresh():
    proj = fsl.ProjThresh()
    assert proj.cmd == 'proj_thresh'
    with pytest.raises(ValueError):
        proj.run()
    proj.inputs.volumes = ['vol1', 'vol2', 'vol3']
    proj.inputs.threshold = 3
    assert proj.cmdline == 'proj_thresh vol1 vol2 vol3 3'
    proj2 = fsl.ProjThresh(threshold=10, volumes=['vola', 'volb'])
    assert proj2.cmdline == 'proj_thresh vola volb 10'
    proj3 = fsl.ProjThresh()
    results = proj3.run(volumes=['inp1', 'inp3', 'inp2'], threshold=2)
    assert results.runtime.cmdline == 'proj_thresh inp1 inp3 inp2 2'
    assert results.runtime.returncode != 0
    assert isinstance(results.interface.inputs.volumes, list)
    assert results.interface.inputs.threshold == 2