import pytest
import sys
from contextlib import contextmanager
from io import StringIO
from ...utils import nipype_cmd
def test_main_returns_2_on_empty(self):
    with pytest.raises(SystemExit) as cm:
        with capture_sys_output() as (stdout, stderr):
            nipype_cmd.main(['nipype_cmd'])
    exit_exception = cm.value
    assert exit_exception.code == 2
    msg = 'usage: nipype_cmd [-h] module interface\nnipype_cmd: error: the following arguments are required: module, interface\n'
    assert stderr.getvalue() == msg
    assert stdout.getvalue() == ''