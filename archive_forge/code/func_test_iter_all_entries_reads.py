import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_iter_all_entries_reads(self):
    self.shrink_page_size()
    builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
    nodes = self.make_nodes(10000, 2, 2)
    for node in nodes:
        builder.add_node(*node)
    t = transport.get_transport_from_url('trace+' + self.get_url(''))
    size = t.put_file('index', builder.finish())
    page_size = btree_index._PAGE_SIZE
    del builder
    index = btree_index.BTreeGraphIndex(t, 'index', size)
    del t._activity[:]
    self.assertEqual([], t._activity)
    found_nodes = self.time(list, index.iter_all_entries())
    bare_nodes = []
    for node in found_nodes:
        self.assertTrue(node[0] is index)
        bare_nodes.append(node[1:])
    self.assertEqual(3, len(index._row_lengths), 'Not enough rows: %r' % index._row_lengths)
    self.assertEqual(20000, len(found_nodes))
    self.assertEqual(set(nodes), set(bare_nodes))
    total_pages = sum(index._row_lengths)
    self.assertEqual(total_pages, index._row_offsets[-1])
    self.assertEqualApproxCompressed(1303220, size)
    first_byte = index._row_offsets[-2] * page_size
    readv_request = []
    for offset in range(first_byte, size, page_size):
        readv_request.append((offset, page_size))
    readv_request[-1] = (readv_request[-1][0], size % page_size)
    expected = [('readv', 'index', [(0, page_size)], False, None), ('readv', 'index', readv_request, False, None)]
    if expected != t._activity:
        self.assertEqualDiff(pprint.pformat(expected), pprint.pformat(t._activity))