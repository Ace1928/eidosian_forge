import os
import nipype.interfaces.fsl.dti as fsl
from nipype.interfaces.fsl import Info, no_fsl
from nipype.interfaces.base import Undefined
import pytest
from nipype.testing.fixtures import create_files_in_directory
@pytest.mark.xfail(reason='These tests are skipped until we clean up some of this code')
def test_Find_the_biggest():
    fbg = fsl.FindTheBiggest()
    assert fbg.cmd == 'find_the_biggest'
    with pytest.raises(ValueError):
        fbg.run()
    fbg.inputs.infiles = 'seed*'
    fbg.inputs.outfile = 'fbgfile'
    assert fbg.cmdline == 'find_the_biggest seed* fbgfile'
    fbg2 = fsl.FindTheBiggest(infiles='seed2*', outfile='fbgfile2')
    assert fbg2.cmdline == 'find_the_biggest seed2* fbgfile2'
    fbg3 = fsl.FindTheBiggest()
    results = fbg3.run(infiles='seed3', outfile='out3')
    assert results.runtime.cmdline == 'find_the_biggest seed3 out3'