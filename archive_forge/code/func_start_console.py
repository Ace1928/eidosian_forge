import os
import shutil
import sys
import tempfile
from subprocess import check_output
from flaky import flaky
import pytest
from traitlets.tests.utils import check_help_all_output
def start_console():
    """Start `jupyter console` using pexpect"""
    import pexpect
    args = ['-m', 'jupyter_console', '--colors=NoColor']
    cmd = sys.executable
    env = os.environ.copy()
    env['JUPYTER_CONSOLE_TEST'] = '1'
    env['PROMPT_TOOLKIT_NO_CPR'] = '1'
    try:
        p = pexpect.spawn(cmd, args=args, env=env)
    except IOError:
        pytest.skip("Couldn't find command %s" % cmd)
    t = 120
    p.expect('In \\[\\d+\\]', timeout=t)
    return (p, pexpect, t)