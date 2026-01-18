from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_get_parent_map_mixed(self):
    """Anything that can be returned from cache, should be"""
    self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
    self.assertEqual([b'b'], self.inst_pp.calls)
    self.assertEqual({b'a': (b'b',)}, self.caching_pp.get_parent_map([b'a', b'b']))
    self.assertEqual([b'b', b'a'], self.inst_pp.calls)