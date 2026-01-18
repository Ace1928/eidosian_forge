from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_content_rename_one_to_one(self):
    b11 = make_object(Blob, data=b'a\nb\nc\nd\n')
    b12 = make_object(Blob, data=b'a\nb\nc\ne\n')
    b21 = make_object(Blob, data=b'e\nf\ng\n\nh')
    b22 = make_object(Blob, data=b'e\nf\ng\n\ni')
    tree1 = self.commit_tree([(b'a', b11), (b'b', b21)])
    tree2 = self.commit_tree([(b'c', b12), (b'd', b22)])
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, b11.id), (b'c', F, b12.id)), TreeChange(CHANGE_RENAME, (b'b', F, b21.id), (b'd', F, b22.id))], self.detect_renames(tree1, tree2))