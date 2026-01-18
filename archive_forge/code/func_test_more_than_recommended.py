import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_more_than_recommended(self):
    index = self.make_index(4096 * 100, 2)
    self.assertExpandOffsets([1, 10], index, [1, 10])
    self.assertExpandOffsets([1, 10, 20], index, [1, 10, 20])