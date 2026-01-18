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
def test_file_given_non_existing_file(self):
    temp_filename = NON_EXISTING_FILE
    obj = {TEST_FILE_KEY: temp_filename}
    t = FileOrData(obj=obj, file_key_name=TEST_FILE_KEY)
    self.expect_exception(t.as_file, 'does not exists')