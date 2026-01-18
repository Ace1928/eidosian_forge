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
def test_get_config_path_env_var(monkeypatch):
    config_path_sentinel = 'config_path'
    monkeypatch.setenv(environment_vars.CLOUD_SDK_CONFIG_DIR, config_path_sentinel)
    config_path = _cloud_sdk.get_config_path()
    assert config_path == config_path_sentinel