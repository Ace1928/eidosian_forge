import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_end_of_merge_not_last_revision_in_branch(self):
    self.assertSortAndIterate({'A': ['B'], 'B': []}, 'A', [(0, 'A', 0, False), (1, 'B', 0, True)], False)
    self.assertSortAndIterate({'A': ['B'], 'B': []}, 'A', [(0, 'A', 0, (2,), False), (1, 'B', 0, (1,), True)], True)