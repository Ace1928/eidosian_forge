from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_filter_changes(self):
    blob_a = make_object(Blob, data=b'a')
    blob_b = make_object(Blob, data=b'b')
    c1 = self.make_linear_commits(1, trees={1: [(b'x/a', blob_a), (b'y/b', blob_b)]})[0]
    walker = Walker(self.store, c1.id)
    walker_entry = next(iter(walker))
    changes = walker_entry.changes(path_prefix=b'x')
    entry_a = (b'a', F, blob_a.id)
    self.assertEqual([TreeChange.add(entry_a)], changes)