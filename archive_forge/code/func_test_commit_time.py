from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from .utils import build_commit_graph, make_object
def test_commit_time(self):
    c1, c2, c3 = build_commit_graph(self.store, [[1], [2, 1], [3, 2]], attrs={1: {'commit_time': 124}, 2: {'commit_time': 123}})
    self.assertEqual(124, c1.commit_time)
    self.assertEqual(123, c2.commit_time)
    self.assertTrue(c2.commit_time < c1.commit_time < c3.commit_time)