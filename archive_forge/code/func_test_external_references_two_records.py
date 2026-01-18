import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_external_references_two_records(self):
    index = self.make_index(ref_lists=1, nodes=[((b'key-1',), b'value', ([(b'key-2',)],)), ((b'key-2',), b'value', ([],))])
    self.assertEqual(set(), index.external_references(0))