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
def test_file_with_custom_dirname(self):
    tempfile = self._create_temp_file(content=TEST_DATA)
    tempfile_dir = os.path.dirname(tempfile)
    tempfile_basename = os.path.basename(tempfile)
    obj = {TEST_FILE_KEY: tempfile_basename}
    t = FileOrData(obj=obj, file_key_name=TEST_FILE_KEY, file_base_path=tempfile_dir)
    self.assertEqual(TEST_DATA, self.get_file_content(t.as_file()))