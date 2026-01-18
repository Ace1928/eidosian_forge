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
def test_get_with_name_on_invalid_object(self):
    self.expect_exception(lambda: self.node['key2'].get_with_name('no-name'), "Expected all values in test_obj/key2 list to have 'name' key")