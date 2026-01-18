from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def assertChangesForMergeEqual(self, expected, parent_trees, merge_tree, **kwargs):
    parent_tree_ids = [t.id for t in parent_trees]
    actual = list(tree_changes_for_merge(self.store, parent_tree_ids, merge_tree.id, **kwargs))
    self.assertEqual(expected, actual)
    parent_tree_ids.reverse()
    expected = [list(reversed(cs)) for cs in expected]
    actual = list(tree_changes_for_merge(self.store, parent_tree_ids, merge_tree.id, **kwargs))
    self.assertEqual(expected, actual)