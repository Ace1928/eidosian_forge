import os
import pytest
import nipype.interfaces.matlab as mlab
@pytest.mark.skipif(no_matlab, reason='matlab is not available')
def test_cmdline():
    default_script_file = clean_workspace_and_get_default_script_file()
    mi = mlab.MatlabCommand(script='whos', script_file='testscript', mfile=False)
    assert mi.cmdline == matlab_cmd + ' -nodesktop -nosplash -singleCompThread -r "fprintf(1,\'Executing code at %s:\\n\',datestr(now));ver,try,whos,catch ME,fprintf(2,\'MATLAB code threw an exception:\\n\');fprintf(2,\'%s\\n\',ME.message);if length(ME.stack) ~= 0, fprintf(2,\'File:%s\\nName:%s\\nLine:%d\\n\',ME.stack.file,ME.stack.name,ME.stack.line);, end;end;;exit"'
    assert mi.inputs.script == 'whos'
    assert mi.inputs.script_file == 'testscript'
    assert not os.path.exists(mi.inputs.script_file), 'scriptfile should not exist'
    assert not os.path.exists(default_script_file), 'default scriptfile should not exist.'