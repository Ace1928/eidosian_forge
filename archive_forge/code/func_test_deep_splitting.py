from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_deep_splitting(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(40)
    chkmap.map((b'aaaaaaaa',), b'v')
    chkmap.map((b'aaaaabaa',), b'v')
    self.assertEqualDiff("'' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n", chkmap._dump_tree())
    chkmap.map((b'aaabaaaa',), b'v')
    chkmap.map((b'aaababaa',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' LeafNode\n      ('aaabaaaa',) 'v'\n      ('aaababaa',) 'v'\n", chkmap._dump_tree())
    chkmap.map((b'aaabacaa',), b'v')
    chkmap.map((b'aaabadaa',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' InternalNode\n    'aaabaa' LeafNode\n      ('aaabaaaa',) 'v'\n    'aaabab' LeafNode\n      ('aaababaa',) 'v'\n    'aaabac' LeafNode\n      ('aaabacaa',) 'v'\n    'aaabad' LeafNode\n      ('aaabadaa',) 'v'\n", chkmap._dump_tree())
    chkmap.map((b'aaababba',), b'val')
    chkmap.map((b'aaababca',), b'val')
    self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' InternalNode\n    'aaabaa' LeafNode\n      ('aaabaaaa',) 'v'\n    'aaabab' InternalNode\n      'aaababa' LeafNode\n      ('aaababaa',) 'v'\n      'aaababb' LeafNode\n      ('aaababba',) 'val'\n      'aaababc' LeafNode\n      ('aaababca',) 'val'\n    'aaabac' LeafNode\n      ('aaabacaa',) 'v'\n    'aaabad' LeafNode\n      ('aaabadaa',) 'v'\n", chkmap._dump_tree())
    chkmap.map((b'aaabDaaa',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aaaa' LeafNode\n      ('aaaaaaaa',) 'v'\n      ('aaaaabaa',) 'v'\n  'aaab' InternalNode\n    'aaabD' LeafNode\n      ('aaabDaaa',) 'v'\n    'aaaba' InternalNode\n      'aaabaa' LeafNode\n      ('aaabaaaa',) 'v'\n      'aaabab' InternalNode\n        'aaababa' LeafNode\n      ('aaababaa',) 'v'\n        'aaababb' LeafNode\n      ('aaababba',) 'val'\n        'aaababc' LeafNode\n      ('aaababca',) 'val'\n      'aaabac' LeafNode\n      ('aaabacaa',) 'v'\n      'aaabad' LeafNode\n      ('aaabadaa',) 'v'\n", chkmap._dump_tree())