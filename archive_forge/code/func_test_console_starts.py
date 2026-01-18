import os
import shutil
import sys
import tempfile
from subprocess import check_output
from flaky import flaky
import pytest
from traitlets.tests.utils import check_help_all_output
@flaky
@pytest.mark.skipif(should_skip, reason='not supported')
def test_console_starts():
    """test that `jupyter console` starts a terminal"""
    p, pexpect, t = start_console()
    p.sendline('5')
    p.expect(['Out\\[\\d+\\]: 5', pexpect.EOF], timeout=t)
    p.expect(['In \\[\\d+\\]', pexpect.EOF], timeout=t)
    stop_console(p, pexpect, t)