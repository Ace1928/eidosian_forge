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
def test_display_text():
    """Ensure display protocol plain/text key is supported"""
    p, pexpect, t = start_console()
    p.sendline('x = %lsmagic')
    p.expect('In \\[\\d+\\]', timeout=t)
    p.sendline('from IPython.display import display; display(x);')
    p.expect('Available line magics:', timeout=t)
    p.expect('In \\[\\d+\\]', timeout=t)
    stop_console(p, pexpect, t)