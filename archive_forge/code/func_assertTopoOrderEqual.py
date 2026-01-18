from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def assertTopoOrderEqual(self, expected_commits, commits):
    entries = [TestWalkEntry(c, None) for c in commits]
    actual_ids = [e.commit.id for e in list(_topo_reorder(entries))]
    self.assertEqual([c.id for c in expected_commits], actual_ids)