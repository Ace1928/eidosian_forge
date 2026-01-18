from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def make_root_only_aaa_ddd_map(self, search_key_func=None):
    return self.get_map({(b'aaa',): b'initial aaa content', (b'ddd',): b'initial ddd content'}, search_key_func=search_key_func)