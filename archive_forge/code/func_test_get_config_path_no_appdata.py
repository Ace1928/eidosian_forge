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
def test_get_config_path_no_appdata(monkeypatch):
    monkeypatch.delenv(_cloud_sdk._WINDOWS_CONFIG_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv('SystemDrive', 'G:')
    config_path = _cloud_sdk.get_config_path()
    assert os.path.split(config_path) == ('G:/\\', _cloud_sdk._CONFIG_DIRECTORY)