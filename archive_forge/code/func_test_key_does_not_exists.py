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
def test_key_does_not_exists(self):
    self.expect_exception(lambda: self.node['not-exists-key'], 'Expected key not-exists-key in test_obj')
    self.expect_exception(lambda: self.node['key3']['not-exists-key'], 'Expected key not-exists-key in test_obj/key3')