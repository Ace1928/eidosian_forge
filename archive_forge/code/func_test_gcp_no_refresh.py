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
def test_gcp_no_refresh(self):
    fake_config = FakeConfig()
    self.assertFalse(hasattr(fake_config, 'get_api_key_with_prefix'))
    KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='gcp', get_google_credentials=lambda: _raise_exception('SHOULD NOT BE CALLED')).load_and_set(fake_config)
    self.assertIsNotNone(fake_config.get_api_key_with_prefix)
    self.assertEqual(TEST_HOST, fake_config.host)
    self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, fake_config.api_key['authorization'])