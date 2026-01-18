from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_changes_one_parent(self):
    blob_a1 = make_object(Blob, data=b'a1')
    blob_a2 = make_object(Blob, data=b'a2')
    blob_b2 = make_object(Blob, data=b'b2')
    c1, c2 = self.make_linear_commits(2, trees={1: [(b'a', blob_a1)], 2: [(b'a', blob_a2), (b'b', blob_b2)]})
    e1 = TestWalkEntry(c1, [TreeChange.add((b'a', F, blob_a1.id))])
    e2 = TestWalkEntry(c2, [TreeChange(CHANGE_MODIFY, (b'a', F, blob_a1.id), (b'a', F, blob_a2.id)), TreeChange.add((b'b', F, blob_b2.id))])
    self.assertWalkYields([e2, e1], [c2.id])