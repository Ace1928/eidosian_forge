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
def test_new_client_from_config(self):
    kubeconfigs = self._create_multi_config()
    client = new_client_from_config(config_file=kubeconfigs, context='simple_token')
    self.assertEqual(TEST_HOST, client.configuration.host)
    self.assertEqual(BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, client.configuration.api_key['authorization'])