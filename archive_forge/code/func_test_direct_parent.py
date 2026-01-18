from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def test_direct_parent(self):
    graph = {'G': ['D', 'F'], 'F': ['E'], 'D': ['C'], 'C': ['B'], 'E': ['B'], 'B': ['A'], 'A': []}
    self.assertEqual(self.run_test(graph, ['G', 'D']), {'D'})