import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def assertSortAndIterate(self, graph, branch_tip, result_list, generate_revno, mainline_revisions=None):
    """Check that merge based sort and iter_topo_order on graph works."""
    value = merge_sort(graph, branch_tip, mainline_revisions=mainline_revisions, generate_revno=generate_revno)
    if result_list != value:
        self.assertEqualDiff(pprint.pformat(result_list), pprint.pformat(value))
    self.assertEqual(result_list, list(MergeSorter(graph, branch_tip, mainline_revisions=mainline_revisions, generate_revno=generate_revno).iter_topo_order()))