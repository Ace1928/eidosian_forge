import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_empty_key_count_no_size(self):
    builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
    t = transport.get_transport_from_url('trace+' + self.get_url(''))
    t.put_file('index', builder.finish())
    index = btree_index.BTreeGraphIndex(t, 'index', None)
    del t._activity[:]
    self.assertEqual([], t._activity)
    self.assertEqual(0, index.key_count())
    self.assertEqual([('get', 'index')], t._activity)