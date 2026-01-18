from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def assertWalkYields(self, expected, *args, **kwargs):
    walker = Walker(self.store, *args, **kwargs)
    expected = list(expected)
    for i, entry in enumerate(expected):
        if isinstance(entry, Commit):
            expected[i] = TestWalkEntry(entry, None)
    actual = list(walker)
    self.assertEqual(expected, actual)