import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(builtins, 'open', mock.mock_open(read_data='{"a": "b"}'))
def test_load_unknown_extension(self):
    fname = 'abc'
    self.assertRaisesRegex(exc.ClientException, 'must have .json or .yaml extension', create_resources.load_from_file, fname)