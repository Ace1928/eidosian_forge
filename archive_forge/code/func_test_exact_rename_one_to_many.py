from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_exact_rename_one_to_many(self):
    blob = make_object(Blob, data=b'1')
    tree1 = self.commit_tree([(b'a', blob)])
    tree2 = self.commit_tree([(b'b', blob), (b'c', blob)])
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'b', F, blob.id)), TreeChange(CHANGE_COPY, (b'a', F, blob.id), (b'c', F, blob.id))], self.detect_renames(tree1, tree2))