from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_have3_want6(self):
    self.assertMissingMatch([self.cmt(3).id], [self.cmt(7).id], [self.cmt(7).id, self.cmt(6).id, self.cmt(4).id, self.cmt(7).tree, self.cmt(6).tree, self.cmt(4).tree, self.f1_4_id])