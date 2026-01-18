import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_dict_recursive(self):
    expected = "a='b', c.1='d', c.2=''"
    self.assertEqual(expected, utils.format_dict({'a': 'b', 'c': {'1': 'd', '2': ''}}))
    self.assertEqual(expected, utils.format_dict({'c': {'1': 'd', '2': ''}, 'a': 'b'}))
    self.assertIsNone(utils.format_dict(None))
    expected = "a1='A', a2.b1.c1='B', a2.b1.c2=, a2.b2='D'"
    self.assertEqual(expected, utils.format_dict({'a1': 'A', 'a2': {'b1': {'c1': 'B', 'c2': None}, 'b2': 'D'}}))
    self.assertEqual(expected, utils.format_dict({'a2': {'b1': {'c2': None, 'c1': 'B'}, 'b2': 'D'}, 'a1': 'A'}))