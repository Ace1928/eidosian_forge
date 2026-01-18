import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_merge_sort_race(self):
    graph = {'A': [], 'B': ['A'], 'C': ['B'], 'D': ['B', 'C'], 'F': ['B', 'D']}
    self.assertSortAndIterate(graph, 'F', [(0, 'F', 0, (3,), False), (1, 'D', 1, (2, 2, 1), False), (2, 'C', 2, (2, 1, 1), True), (3, 'B', 0, (2,), False), (4, 'A', 0, (1,), True)], True)
    graph = {'A': [], 'B': ['A'], 'C': ['B'], 'X': ['B'], 'D': ['X', 'C'], 'F': ['B', 'D']}
    self.assertSortAndIterate(graph, 'F', [(0, 'F', 0, (3,), False), (1, 'D', 1, (2, 1, 2), False), (2, 'C', 2, (2, 2, 1), True), (3, 'X', 1, (2, 1, 1), True), (4, 'B', 0, (2,), False), (5, 'A', 0, (1,), True)], True)