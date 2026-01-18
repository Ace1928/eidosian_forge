from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__dump_tree_in_progress(self):
    chkmap = self._get_map({(b'aaa',): b'value1', (b'aab',): b'value2'}, maximum_size=10)
    chkmap.map((b'bbb',), b'value3')
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'value1'\n    'aab' LeafNode\n      ('aab',) 'value2'\n  'b' LeafNode\n      ('bbb',) 'value3'\n", chkmap._dump_tree())
    self.assertEqualDiff("'' InternalNode None\n  'a' InternalNode sha1:6b0d881dd739a66f733c178b24da64395edfaafd\n    'aaa' LeafNode sha1:40b39a08d895babce17b20ae5f62d187eaa4f63a\n      ('aaa',) 'value1'\n    'aab' LeafNode sha1:ad1dc7c4e801302c95bf1ba7b20bc45e548cd51a\n      ('aab',) 'value2'\n  'b' LeafNode None\n      ('bbb',) 'value3'\n", chkmap._dump_tree(include_keys=True))