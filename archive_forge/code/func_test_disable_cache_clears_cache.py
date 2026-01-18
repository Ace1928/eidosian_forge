from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_disable_cache_clears_cache(self):
    self.caching_pp.get_parent_map([b'rev1'])
    self.assertEqual(2, len(self.caching_pp._cache))
    self.caching_pp.disable_cache()
    self.assertIs(None, self.caching_pp._cache)