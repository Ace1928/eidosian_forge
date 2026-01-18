from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_empty_ancestry(self):
    self.assertSearchResult([], [], 0, {}, (), [b'tip-rev-id'], 10)