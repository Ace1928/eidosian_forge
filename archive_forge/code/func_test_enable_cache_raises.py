from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_enable_cache_raises(self):
    e = self.assertRaises(AssertionError, self.caching_pp.enable_cache)
    self.assertEqual('Cache enabled when already enabled.', str(e))