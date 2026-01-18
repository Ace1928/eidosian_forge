import os.path
import tempfile
import yaml
from mistralclient import utils
from oslo_serialization import jsonutils
from oslotest import base
def test_load_yaml_content(self):
    self.assertDictEqual(ENV_DICT, utils.load_content(ENV_YAML))