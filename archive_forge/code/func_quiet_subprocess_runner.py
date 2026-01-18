import json
import os
import sys
import tempfile
from contextlib import contextmanager
from os.path import abspath
from os.path import join as pjoin
from subprocess import STDOUT, check_call, check_output
from ._in_process import _in_proc_script_path
def quiet_subprocess_runner(cmd, cwd=None, extra_environ=None):
    """Call the subprocess while suppressing output.

    This uses :func:`subprocess.check_output` under the hood.
    """
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)
    check_output(cmd, cwd=cwd, env=env, stderr=STDOUT)