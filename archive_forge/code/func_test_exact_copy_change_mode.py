from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_exact_copy_change_mode(self):
    blob = make_object(Blob, data=b'a\nb\nc\nd\n')
    tree1 = self.commit_tree([(b'a', blob)])
    tree2 = self.commit_tree([(b'a', blob, 33261), (b'b', blob)])
    self.assertEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob.id), (b'a', 33261, blob.id)), TreeChange(CHANGE_COPY, (b'a', F, blob.id), (b'b', F, blob.id))], self.detect_renames(tree1, tree2))