import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def assertMultiBisectRight(self, offsets, search_keys, fixed_keys):
    self.assertEqual(offsets, btree_index.BTreeGraphIndex._multi_bisect_right(search_keys, fixed_keys))