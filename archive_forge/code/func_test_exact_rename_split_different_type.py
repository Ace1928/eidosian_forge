from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_exact_rename_split_different_type(self):
    blob = make_object(Blob, data=b'/foo')
    tree1 = self.commit_tree([(b'a', blob, 33188)])
    tree2 = self.commit_tree([(b'a', blob, 40960)])
    self.assertEqual([TreeChange.add((b'a', 40960, blob.id)), TreeChange.delete((b'a', 33188, blob.id))], self.detect_renames(tree1, tree2))