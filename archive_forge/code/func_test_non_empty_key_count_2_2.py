import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_non_empty_key_count_2_2(self):
    builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
    nodes = self.make_nodes(35, 2, 2)
    for node in nodes:
        builder.add_node(*node)
    t = transport.get_transport_from_url('trace+' + self.get_url(''))
    size = t.put_file('index', builder.finish())
    index = btree_index.BTreeGraphIndex(t, 'index', size)
    del t._activity[:]
    self.assertEqual([], t._activity)
    self.assertEqual(70, index.key_count())
    self.assertEqual([('readv', 'index', [(0, size)], False, None)], t._activity)
    self.assertEqualApproxCompressed(1173, size)