from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_ancestry_1(self):
    self.assertSearchResult([b'rev4'], [b'rev1'], 4, ancestry_1, (), [b'rev1'], 10)
    self.assertSearchResult([b'rev2a', b'rev2b'], [b'rev1'], 2, ancestry_1, (), [b'rev1'], 1)