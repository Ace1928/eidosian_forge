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
def test_file_given_file(self):
    temp_filename = _create_temp_file_with_content(TEST_DATA)
    obj = {TEST_FILE_KEY: temp_filename}
    t = FileOrData(obj=obj, file_key_name=TEST_FILE_KEY)
    self.assertEqual(TEST_DATA, self.get_file_content(t.as_file()))