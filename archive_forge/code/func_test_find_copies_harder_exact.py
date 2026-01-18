from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_find_copies_harder_exact(self):
    blob = make_object(Blob, data=b'blob')
    tree1 = self.commit_tree([(b'a', blob)])
    tree2 = self.commit_tree([(b'a', blob), (b'b', blob)])
    self.assertEqual([TreeChange.add((b'b', F, blob.id))], self.detect_renames(tree1, tree2))
    self.assertEqual([TreeChange(CHANGE_COPY, (b'a', F, blob.id), (b'b', F, blob.id))], self.detect_renames(tree1, tree2, find_copies_harder=True))