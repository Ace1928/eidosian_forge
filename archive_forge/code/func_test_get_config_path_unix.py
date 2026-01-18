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
@mock.patch('os.path.expanduser')
def test_get_config_path_unix(expanduser):
    expanduser.side_effect = lambda path: path
    config_path = _cloud_sdk.get_config_path()
    assert os.path.split(config_path) == ('~/.config', _cloud_sdk._CONFIG_DIRECTORY)