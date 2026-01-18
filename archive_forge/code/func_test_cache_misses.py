from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_cache_misses(self):
    self.caching_pp.get_parent_map([b'rev3'])
    self.caching_pp.get_parent_map([b'rev3'])
    self.assertEqual([b'rev3'], self.inst_pp.calls)