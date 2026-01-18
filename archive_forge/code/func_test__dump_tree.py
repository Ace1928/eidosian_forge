from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test__dump_tree(self):
    chkmap = self._get_map({(b'aaa',): b'value1', (b'aab',): b'value2', (b'bbb',): b'value3'}, maximum_size=15)
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'value1'\n    'aab' LeafNode\n      ('aab',) 'value2'\n  'b' LeafNode\n      ('bbb',) 'value3'\n", chkmap._dump_tree())
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'value1'\n    'aab' LeafNode\n      ('aab',) 'value2'\n  'b' LeafNode\n      ('bbb',) 'value3'\n", chkmap._dump_tree())
    self.assertEqualDiff("'' InternalNode sha1:0690d471eb0a624f359797d0ee4672bd68f4e236\n  'a' InternalNode sha1:1514c35503da9418d8fd90c1bed553077cb53673\n    'aaa' LeafNode sha1:4cc5970454d40b4ce297a7f13ddb76f63b88fefb\n      ('aaa',) 'value1'\n    'aab' LeafNode sha1:1d68bc90914ef8a3edbcc8bb28b00cb4fea4b5e2\n      ('aab',) 'value2'\n  'b' LeafNode sha1:3686831435b5596515353364eab0399dc45d49e7\n      ('bbb',) 'value3'\n", chkmap._dump_tree(include_keys=True))