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
def test_data_given_file(self):
    obj = {TEST_FILE_KEY: self._create_temp_file(content=TEST_DATA)}
    t = FileOrData(obj=obj, file_key_name=TEST_FILE_KEY)
    self.assertEqual(TEST_DATA_BASE64, t.as_data())