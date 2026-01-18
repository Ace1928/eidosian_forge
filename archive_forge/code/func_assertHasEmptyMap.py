from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def assertHasEmptyMap(self, chk_bytes):
    empty_leaf_bytes = b'chkleaf:\n0\n1\n0\n\n'
    empty_sha1 = osutils.sha_string(empty_leaf_bytes)
    self.assertEqual(b'8571e09bf1bcc5b9621ce31b3d4c93d6e9a1ed26', empty_sha1)
    root_key = (b'sha1:' + empty_sha1,)
    self.assertEqual(empty_leaf_bytes, self.read_bytes(chk_bytes, root_key))
    return root_key