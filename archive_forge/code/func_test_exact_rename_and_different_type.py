from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_exact_rename_and_different_type(self):
    blob1 = make_object(Blob, data=b'1')
    blob2 = make_object(Blob, data=b'2')
    tree1 = self.commit_tree([(b'a', blob1)])
    tree2 = self.commit_tree([(b'a', blob2, 40960), (b'b', blob1)])
    self.assertEqual([TreeChange.add((b'a', 40960, blob2.id)), TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob1.id))], self.detect_renames(tree1, tree2))