import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_get_item_properties_with_format_func(self):
    formatters = {'attr': utils.format_list}
    res_attr = self._test_get_item_properties_with_formatter(formatters)
    self.assertEqual(utils.format_list(['a', 'b']), res_attr)