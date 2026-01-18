from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_content_rename_max_files(self):
    blob1 = make_object(Blob, data=b'a\nb\nc\nd')
    blob4 = make_object(Blob, data=b'a\nb\nc\ne\n')
    blob2 = make_object(Blob, data=b'e\nf\ng\nh\n')
    blob3 = make_object(Blob, data=b'e\nf\ng\ni\n')
    tree1 = self.commit_tree([(b'a', blob1), (b'b', blob2)])
    tree2 = self.commit_tree([(b'c', blob3), (b'd', blob4)])
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'd', F, blob4.id)), TreeChange(CHANGE_RENAME, (b'b', F, blob2.id), (b'c', F, blob3.id))], self.detect_renames(tree1, tree2))
    self.assertEqual([TreeChange.delete((b'a', F, blob1.id)), TreeChange.delete((b'b', F, blob2.id)), TreeChange.add((b'c', F, blob3.id)), TreeChange.add((b'd', F, blob4.id))], self.detect_renames(tree1, tree2, max_files=1))