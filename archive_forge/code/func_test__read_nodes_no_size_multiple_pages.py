import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test__read_nodes_no_size_multiple_pages(self):
    index = self.make_index(2, 2, nodes=self.make_nodes(160, 2, 2))
    index.key_count()
    num_pages = index._row_offsets[-1]
    trans = transport.get_transport_from_url('trace+' + self.get_url())
    index = btree_index.BTreeGraphIndex(trans, 'index', None)
    del trans._activity[:]
    nodes = dict(index._read_nodes([0]))
    self.assertEqual(list(range(num_pages)), sorted(nodes))