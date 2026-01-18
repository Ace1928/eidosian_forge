import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def make_1000_node_index(self):
    index = self.make_index(4096 * 1000, 6)
    self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=90000, row_lengths=[1, 9, 990], cached_offsets=[0, 5, 500])
    return index