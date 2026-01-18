import os.path
import tempfile
import yaml
from mistralclient import utils
from oslo_serialization import jsonutils
from oslotest import base
def test_load_json(self):
    with tempfile.NamedTemporaryFile() as f:
        f.write(ENV_STR.encode('utf-8'))
        f.flush()
        self.assertDictEqual(ENV_DICT, utils.load_json(f.name))
    self.assertDictEqual(ENV_DICT, utils.load_json(ENV_STR))