from unittest import TestCase
import textwrap
from jsonschema import exceptions
from jsonschema.validators import _LATEST_VERSION
def test_multiple_item_paths(self):
    self.assertShows("\n            Failed validating 'type' in schema['items'][0]:\n                {'type': 'string'}\n\n            On instance[0]['a']:\n                5\n            ", path=[0, 'a'], schema_path=['items', 0, 1])