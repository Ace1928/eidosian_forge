from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_have6_want7(self):
    self.assertMissingMatch([self.cmt(6).id], [self.cmt(7).id], [self.cmt(7).id, self.cmt(7).tree, self.f1_7_id])