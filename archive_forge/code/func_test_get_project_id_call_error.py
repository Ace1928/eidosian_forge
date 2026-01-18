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
@mock.patch('subprocess.check_output', autospec=True, side_effect=subprocess.CalledProcessError(-1, 'testing'))
def test_get_project_id_call_error(check_output):
    project_id = _cloud_sdk.get_project_id()
    assert project_id is None
    assert check_output.called