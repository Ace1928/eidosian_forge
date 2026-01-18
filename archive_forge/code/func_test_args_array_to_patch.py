import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_args_array_to_patch(self):
    my_args = {'attributes': ['str=foo', 'int=1', 'bool=true', 'list=[1, 2, 3]', 'dict={"foo": "bar"}'], 'op': 'add'}
    patch = utils.args_array_to_patch(my_args['op'], my_args['attributes'])
    self.assertEqual([{'op': 'add', 'value': 'foo', 'path': '/str'}, {'op': 'add', 'value': 1, 'path': '/int'}, {'op': 'add', 'value': True, 'path': '/bool'}, {'op': 'add', 'value': [1, 2, 3], 'path': '/list'}, {'op': 'add', 'value': {'foo': 'bar'}, 'path': '/dict'}], patch)