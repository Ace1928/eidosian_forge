from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def test_no_common_ancestor(self):
    graph = {'4': ['2'], '3': ['1'], '2': [], '1': ['0'], '0': []}
    self.assertEqual(self.run_test(graph, ['4', '3']), set())