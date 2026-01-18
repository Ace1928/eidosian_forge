import os
import pytest
import nipype.interfaces.matlab as mlab
@pytest.mark.skipif(no_matlab, reason='matlab is not available')
def test_mlab_init():
    default_script_file = clean_workspace_and_get_default_script_file()
    assert mlab.MatlabCommand._cmd == 'matlab'
    assert mlab.MatlabCommand.input_spec == mlab.MatlabInputSpec
    assert mlab.MatlabCommand().cmd == matlab_cmd
    mc = mlab.MatlabCommand(matlab_cmd='foo_m')
    assert mc.cmd == 'foo_m'