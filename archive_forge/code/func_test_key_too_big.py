import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_key_too_big(self):
    bigKey = b''.join((b'%d' % n for n in range(btree_index._PAGE_SIZE)))
    self.assertRaises(_mod_index.BadIndexKey, self.make_index, nodes=[((bigKey,), b'value', ())])