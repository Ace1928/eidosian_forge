import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_dict_of_list(self):
    expected = 'a=a1, a2; b=b1, b2; c=c1, c2; e='
    self.assertEqual(expected, utils.format_dict_of_list({'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'c': ['c1', 'c2'], 'd': None, 'e': []}))
    self.assertEqual(expected, utils.format_dict_of_list({'c': ['c1', 'c2'], 'a': ['a2', 'a1'], 'b': ['b2', 'b1'], 'e': []}))
    self.assertIsNone(utils.format_dict_of_list(None))