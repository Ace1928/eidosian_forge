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
@mock.patch('os.name', new='nt')
def test_get_config_path_windows(monkeypatch):
    appdata = 'appdata'
    monkeypatch.setenv(_cloud_sdk._WINDOWS_CONFIG_ROOT_ENV_VAR, appdata)
    config_path = _cloud_sdk.get_config_path()
    assert os.path.split(config_path) == (appdata, _cloud_sdk._CONFIG_DIRECTORY)