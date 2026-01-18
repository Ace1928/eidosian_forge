import json
import six
from os_service_types import data
from os_service_types.tests import base
def test_load_non_existing(self):
    self.assertRaises(FileNotFoundError, data.read_data, '/non-existing-file')