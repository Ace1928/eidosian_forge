from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_extras_using_cached(self):
    self.assertEqual({}, self.caching_pp.get_cached_parent_map([b'rev3']))
    self.assertEqual({}, self.caching_pp.get_parent_map([b'rev3']))
    self.assertEqual({b'rev2': [b'rev1']}, self.caching_pp.get_cached_parent_map([b'rev2']))
    self.assertEqual([b'rev3'], self.inst_pp.calls)