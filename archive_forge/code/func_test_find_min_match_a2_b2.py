import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_min_match_a2_b2(self):
    items = self._get_test_items()
    sort_str = 'b'
    flair = {'a': 2, 'b': 2}
    expect_items = [items[2]]
    self.assertEqual(expect_items, utils.find_min_match(items, sort_str, **flair))