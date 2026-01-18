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
def test_get_with_name_on_non_list_object(self):
    self.expect_exception(lambda: self.node['key3'].get_with_name('no-name'), 'Expected test_obj/key3 to be a list')