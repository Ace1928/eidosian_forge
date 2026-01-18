from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_common_ancestor_two_repos(self):
    """Ensure we do unique_lca using data from two repos"""
    mainline_tree = self.prepare_memory_tree('mainline')
    self.build_ancestry(mainline_tree, mainline)
    self.addCleanup(mainline_tree.unlock)
    feature_tree = self.prepare_memory_tree('feature')
    self.build_ancestry(feature_tree, feature_branch)
    self.addCleanup(feature_tree.unlock)
    graph = mainline_tree.branch.repository.get_graph(feature_tree.branch.repository)
    self.assertEqual(b'rev2b', graph.find_unique_lca(b'rev2a', b'rev3b'))