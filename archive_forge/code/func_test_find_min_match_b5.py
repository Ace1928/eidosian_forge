import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_find_min_match_b5(self):
    items = self._get_test_items()
    sort_str = 'b'
    flair = {'b': 5}
    expect_items = []
    self.assertEqual(expect_items, utils.find_min_match(items, sort_str, **flair))