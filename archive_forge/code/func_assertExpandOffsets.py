import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def assertExpandOffsets(self, expected, index, offsets):
    self.assertEqual(expected, index._expand_offsets(offsets), 'We did not get the expected value after expanding %s' % (offsets,))