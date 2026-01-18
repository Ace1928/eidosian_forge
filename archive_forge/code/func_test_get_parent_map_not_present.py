from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_get_parent_map_not_present(self):
    """The cache should also track when a revision doesn't exist"""
    self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
    self.assertEqual([b'b'], self.inst_pp.calls)
    self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
    self.assertEqual([b'b'], self.inst_pp.calls)