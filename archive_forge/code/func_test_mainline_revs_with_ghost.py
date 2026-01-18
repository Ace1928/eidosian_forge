import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_mainline_revs_with_ghost(self):
    self.assertSortAndIterate({'B': [], 'C': ['B']}.items(), 'C', [(0, 'C', 0, (2,), False), (1, 'B', 0, (1,), True)], True, mainline_revisions=['A', 'B', 'C'])