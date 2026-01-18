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
def test_load_kube_config(self):
    expected = FakeConfig(host=TEST_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64)
    config_file = self._create_temp_file(yaml.safe_dump(self.TEST_KUBE_CONFIG))
    actual = FakeConfig()
    load_kube_config(config_file=config_file, context='simple_token', client_configuration=actual)
    self.assertEqual(expected, actual)