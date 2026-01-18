import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_list_of_dicts(self):
    expected = "a='b', c='d'\ne='f'"
    sorted_data = [{'a': 'b', 'c': 'd'}, {'e': 'f'}]
    unsorted_data = [{'c': 'd', 'a': 'b'}, {'e': 'f'}]
    self.assertEqual(expected, utils.format_list_of_dicts(sorted_data))
    self.assertEqual(expected, utils.format_list_of_dicts(unsorted_data))
    self.assertEqual('', utils.format_list_of_dicts([]))
    self.assertEqual('', utils.format_list_of_dicts([{}]))
    self.assertIsNone(utils.format_list_of_dicts(None))