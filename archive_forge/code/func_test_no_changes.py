from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_no_changes(self):
    self.assertMissingMatch([self.cmt(3).id], [self.cmt(3).id], [])