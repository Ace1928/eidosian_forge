from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_content_rename_gitlink(self):
    blob1 = make_object(Blob, data=b'blob1')
    blob2 = make_object(Blob, data=b'blob2')
    link1 = b'1' * 40
    link2 = b'2' * 40
    tree1 = self.commit_tree([(b'a', blob1), (b'b', link1, 57344)])
    tree2 = self.commit_tree([(b'c', blob2), (b'd', link2, 57344)])
    self.assertEqual([TreeChange.delete((b'a', 33188, blob1.id)), TreeChange.delete((b'b', 57344, link1)), TreeChange.add((b'c', 33188, blob2.id)), TreeChange.add((b'd', 57344, link2))], self.detect_renames(tree1, tree2))