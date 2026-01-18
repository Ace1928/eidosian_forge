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
def test_current_context(self):
    loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG)
    expected_contexts = ConfigNode('', self.TEST_KUBE_CONFIG)['contexts']
    self.assertEqual(expected_contexts.get_with_name('no_user').value, loader.current_context)