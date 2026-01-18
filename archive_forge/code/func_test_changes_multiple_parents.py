from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_changes_multiple_parents(self):
    blob_a1 = make_object(Blob, data=b'a1')
    blob_b2 = make_object(Blob, data=b'b2')
    blob_a3 = make_object(Blob, data=b'a3')
    c1, c2, c3 = self.make_commits([[1], [2], [3, 1, 2]], trees={1: [(b'a', blob_a1)], 2: [(b'b', blob_b2)], 3: [(b'a', blob_a3), (b'b', blob_b2)]})
    changes = [[TreeChange(CHANGE_MODIFY, (b'a', F, blob_a1.id), (b'a', F, blob_a3.id)), TreeChange.add((b'a', F, blob_a3.id))]]
    self.assertWalkYields([TestWalkEntry(c3, changes)], [c3.id], exclude=[c1.id, c2.id])