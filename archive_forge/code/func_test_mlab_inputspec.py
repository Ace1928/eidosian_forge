import os
import pytest
import nipype.interfaces.matlab as mlab
@pytest.mark.skipif(no_matlab, reason='matlab is not available')
def test_mlab_inputspec():
    default_script_file = clean_workspace_and_get_default_script_file()
    spec = mlab.MatlabInputSpec()
    for k in ['paths', 'script', 'nosplash', 'mfile', 'logfile', 'script_file', 'nodesktop']:
        assert k in spec.copyable_trait_names()
    assert spec.nodesktop
    assert spec.nosplash
    assert spec.mfile
    assert spec.script_file == default_script_file