from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_no_cache_misses(self):
    self.caching_pp.disable_cache()
    self.caching_pp.enable_cache(cache_misses=False)
    self.caching_pp.get_parent_map([b'rev3'])
    self.caching_pp.get_parent_map([b'rev3'])
    self.assertEqual([b'rev3', b'rev3'], self.inst_pp.calls)