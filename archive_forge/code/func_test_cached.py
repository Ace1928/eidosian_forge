from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_cached(self):
    self.assertEqual({b'rev1': []}, self.caching_pp.get_parent_map([b'rev1']))
    self.assertEqual([b'rev1'], self.inst_pp.calls)
    self.assertEqual({b'rev1': [], b'rev2': [b'rev1']}, self.caching_pp._cache)
    self.assertEqual({b'rev1': []}, self.caching_pp.get_parent_map([b'rev1']))
    self.assertEqual([b'rev1'], self.inst_pp.calls)