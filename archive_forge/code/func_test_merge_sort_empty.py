import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_merge_sort_empty(self):
    self.assertSortAndIterate({}, None, [], False)
    self.assertSortAndIterate({}, None, [], True)
    self.assertSortAndIterate({}, NULL_REVISION, [], False)
    self.assertSortAndIterate({}, NULL_REVISION, [], True)