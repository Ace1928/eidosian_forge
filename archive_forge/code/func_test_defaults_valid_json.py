import glob
import json
import os
import jsonschema
from testtools import content
from openstack.config import defaults
from openstack.tests.unit.config import base
def test_defaults_valid_json(self):
    _schema_path = os.path.join(os.path.dirname(os.path.realpath(defaults.__file__)), 'schema.json')
    with open(_schema_path, 'r') as f:
        schema = json.load(f)
    self.validator = jsonschema.Draft4Validator(schema)
    self.addOnException(self.json_diagnostics)
    self.filename = os.path.join(os.path.dirname(os.path.realpath(defaults.__file__)), 'defaults.json')
    with open(self.filename, 'r') as f:
        self.json_data = json.load(f)
    self.assertTrue(self.validator.is_valid(self.json_data))