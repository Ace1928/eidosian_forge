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
def test_gcp_get_api_key_with_prefix(self):

    class cred_old:
        token = TEST_DATA_BASE64
        expiry = DATETIME_EXPIRY_PAST

    class cred_new:
        token = TEST_ANOTHER_DATA_BASE64
        expiry = DATETIME_EXPIRY_FUTURE
    fake_config = FakeConfig()
    _get_google_credentials = mock.Mock()
    _get_google_credentials.side_effect = [cred_old, cred_new]
    loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='expired_gcp_refresh', get_google_credentials=_get_google_credentials)
    loader.load_and_set(fake_config)
    original_expiry = _get_expiry(loader, 'expired_gcp_refresh')
    token = fake_config.get_api_key_with_prefix()
    new_expiry = _get_expiry(loader, 'expired_gcp_refresh')
    self.assertTrue(new_expiry > original_expiry)
    self.assertEqual(BEARER_TOKEN_FORMAT % TEST_ANOTHER_DATA_BASE64, loader.token)
    self.assertEqual(BEARER_TOKEN_FORMAT % TEST_ANOTHER_DATA_BASE64, token)