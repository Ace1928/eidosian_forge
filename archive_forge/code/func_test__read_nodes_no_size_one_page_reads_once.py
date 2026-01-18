import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__read_nodes_no_size_one_page_reads_once(self):
    self.make_index(nodes=[((b'key',), b'value', ())])
    trans = transport.get_transport_from_url('trace+' + self.get_url())
    index = btree_index.BTreeGraphIndex(trans, 'index', None)
    del trans._activity[:]
    nodes = dict(index._read_nodes([0]))
    self.assertEqual({0}, set(nodes))
    node = nodes[0]
    self.assertEqual([(b'key',)], node.all_keys())
    self.assertEqual([('get', 'index')], trans._activity)