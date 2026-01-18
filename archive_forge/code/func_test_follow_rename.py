from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_follow_rename(self):
    blob = make_object(Blob, data=b'blob')
    names = [b'a', b'a', b'b', b'b', b'c', b'c']
    trees = {i + 1: [(n, blob, F)] for i, n in enumerate(names)}
    c1, c2, c3, c4, c5, c6 = self.make_linear_commits(6, trees=trees)
    self.assertWalkYields([c5], [c6.id], paths=[b'c'])

    def e(n):
        return (n, F, blob.id)
    self.assertWalkYields([TestWalkEntry(c5, [TreeChange(CHANGE_RENAME, e(b'b'), e(b'c'))]), TestWalkEntry(c3, [TreeChange(CHANGE_RENAME, e(b'a'), e(b'b'))]), TestWalkEntry(c1, [TreeChange.add(e(b'a'))])], [c6.id], paths=[b'c'], follow=True)