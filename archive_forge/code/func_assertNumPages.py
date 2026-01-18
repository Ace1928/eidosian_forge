import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def assertNumPages(self, expected_pages, index, size):
    index._size = size
    self.assertEqual(expected_pages, index._compute_total_pages_in_index())