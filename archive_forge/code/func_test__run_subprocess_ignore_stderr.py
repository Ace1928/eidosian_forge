import io
import json
import os
import subprocess
import sys
import mock
import pytest  # type: ignore
from google.auth import _cloud_sdk
from google.auth import environment_vars
from google.auth import exceptions
def test__run_subprocess_ignore_stderr():
    command = [sys.executable, '-c', 'from __future__ import print_function;' + 'import sys;' + "print('error', file=sys.stderr);" + "print('output', file=sys.stdout)"]
    output = _cloud_sdk._run_subprocess_ignore_stderr(command)
    assert output == b'output\n'
    output = subprocess.check_output(command, stderr=subprocess.STDOUT)
    assert output == b'output\nerror\n' or output == b'error\noutput\n'