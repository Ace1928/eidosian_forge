from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from .utils import build_commit_graph, make_object
def test_missing_parent(self):
    self.assertRaises(ValueError, build_commit_graph, self.store, [[1], [3, 2], [2, 1]])