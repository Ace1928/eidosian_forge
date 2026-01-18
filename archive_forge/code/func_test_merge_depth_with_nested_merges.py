import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_merge_depth_with_nested_merges(self):
    self.assertSortAndIterate({'A': ['D', 'B'], 'B': ['C', 'F'], 'C': ['H'], 'D': ['H', 'E'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}.items(), 'A', [(0, 'A', 0, False), (1, 'B', 1, False), (2, 'C', 1, True), (3, 'D', 0, False), (4, 'E', 1, False), (5, 'F', 2, True), (6, 'G', 1, True), (7, 'H', 0, True)], False)
    self.assertSortAndIterate({'A': ['D', 'B'], 'B': ['C', 'F'], 'C': ['H'], 'D': ['H', 'E'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}.items(), 'A', [(0, 'A', 0, (3,), False), (1, 'B', 1, (1, 3, 2), False), (2, 'C', 1, (1, 3, 1), True), (3, 'D', 0, (2,), False), (4, 'E', 1, (1, 1, 2), False), (5, 'F', 2, (1, 2, 1), True), (6, 'G', 1, (1, 1, 1), True), (7, 'H', 0, (1,), True)], True)