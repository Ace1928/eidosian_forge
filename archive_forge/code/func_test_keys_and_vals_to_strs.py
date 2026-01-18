import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_keys_and_vals_to_strs(self):
    dict_in = {'a': '1', 'b': {'x': 1, 'y': '2', 'z': '3'}, 'c': 7}
    dict_exp = collections.OrderedDict([('a', '1'), ('b', collections.OrderedDict([('x', 1), ('y', '2'), ('z', '3')])), ('c', 7)])
    dict_out = cliutils.keys_and_vals_to_strs(dict_in)
    dict_act = collections.OrderedDict([('a', dict_out['a']), ('b', collections.OrderedDict(sorted(dict_out['b'].items()))), ('c', dict_out['c'])])
    self.assertEqual(str(dict_exp), str(dict_act))