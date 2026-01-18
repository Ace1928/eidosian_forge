from breezy import tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def test_get_known_graph_ancestry_stacked(self):
    """get_known_graph_ancestry works correctly on stacking.

        See <https://bugs.launchpad.net/bugs/715000>.
        """
    branch_a, branch_b, branch_c, revid_1 = self.make_double_stacked_branches()
    for br in [branch_a, branch_b, branch_c]:
        self.assertEqual([revid_1], br.repository.get_known_graph_ancestry([revid_1]).topo_sort())