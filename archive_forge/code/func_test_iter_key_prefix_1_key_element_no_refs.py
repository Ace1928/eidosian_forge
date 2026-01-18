import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_iter_key_prefix_1_key_element_no_refs(self):
    index = self.make_index(nodes=[((b'name',), b'data', ()), ((b'ref',), b'refdata', ())])
    self.assertEqual({(index, (b'name',), b'data'), (index, (b'ref',), b'refdata')}, set(index.iter_entries_prefix([(b'name',), (b'ref',)])))