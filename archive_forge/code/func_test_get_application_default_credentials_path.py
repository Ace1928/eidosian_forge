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
@mock.patch('google.auth._cloud_sdk.get_config_path', autospec=True)
def test_get_application_default_credentials_path(get_config_dir):
    config_path = 'config_path'
    get_config_dir.return_value = config_path
    credentials_path = _cloud_sdk.get_application_default_credentials_path()
    assert credentials_path == os.path.join(config_path, _cloud_sdk._CREDENTIALS_FILENAME)