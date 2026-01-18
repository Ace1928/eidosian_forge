from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
def test_have1_want6(self):
    self.assertMissingMatch([self.cmt(1).id], [self.cmt(6).id], [self.cmt(6).id, self.cmt(4).id, self.cmt(3).id, self.cmt(2).id, self.cmt(6).tree, self.cmt(4).tree, self.cmt(3).tree, self.cmt(2).tree, self.f1_2_id, self.f1_4_id, self.f2_3_id, self.f3_3_id])