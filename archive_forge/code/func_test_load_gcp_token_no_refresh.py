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
def test_load_gcp_token_no_refresh(self):
    loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='gcp', get_google_credentials=lambda: _raise_exception('SHOULD NOT BE CALLED'))
    self.assertTrue(loader._load_auth_provider_token())
    self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, loader.token)