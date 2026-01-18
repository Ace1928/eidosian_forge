import base64
import datetime
import json
import os
import shutil
import tempfile
import unittest
import mock
from ruamel import yaml
from six import PY3, next
from kubernetes.client import Configuration
from .config_exception import ConfigException
from .kube_config import (ENV_KUBECONFIG_PATH_SEPARATOR, ConfigNode, FileOrData,
def test_auth_settings_calls_get_api_key_with_prefix(self):
    expected_token = 'expected_token'

    def fake_get_api_key_with_prefix(identifier):
        self.assertEqual('authorization', identifier)
        return expected_token
    config = Configuration()
    config.get_api_key_with_prefix = fake_get_api_key_with_prefix
    self.assertEqual(expected_token, config.auth_settings()['BearerToken']['value'])