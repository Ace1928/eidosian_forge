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
def test_load_gcp_token_with_refresh(self):

    def cred():
        return None
    cred.token = TEST_ANOTHER_DATA_BASE64
    cred.expiry = datetime.datetime.utcnow()
    loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='expired_gcp', get_google_credentials=lambda: cred)
    original_expiry = _get_expiry(loader, 'expired_gcp')
    self.assertTrue(loader._load_auth_provider_token())
    new_expiry = _get_expiry(loader, 'expired_gcp')
    self.assertTrue(new_expiry > original_expiry)
    self.assertEqual(BEARER_TOKEN_FORMAT % TEST_ANOTHER_DATA_BASE64, loader.token)