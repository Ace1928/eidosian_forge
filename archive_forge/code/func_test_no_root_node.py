import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_no_root_node(self):
    index = self.make_index(4096 * 10, 5)
    self.assertExpandOffsets([0], index, [0])