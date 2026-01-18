import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_LeafNode_2_2(self):
    node_bytes = b'type=leaf\n00\x0000\x00\t00\x00ref00\x00value:0\n00\x0011\x0000\x00ref00\t00\x00ref00\r01\x00ref01\x00value:1\n11\x0033\x0011\x00ref22\t11\x00ref22\r11\x00ref22\x00value:3\n11\x0044\x00\t11\x00ref00\x00value:4\n'
    node = btree_index._LeafNode(node_bytes, 2, 2)
    self.assertEqual({(b'00', b'00'): (b'value:0', ((), ((b'00', b'ref00'),))), (b'00', b'11'): (b'value:1', (((b'00', b'ref00'),), ((b'00', b'ref00'), (b'01', b'ref01')))), (b'11', b'33'): (b'value:3', (((b'11', b'ref22'),), ((b'11', b'ref22'), (b'11', b'ref22')))), (b'11', b'44'): (b'value:4', ((), ((b'11', b'ref00'),)))}, dict(node.all_items()))