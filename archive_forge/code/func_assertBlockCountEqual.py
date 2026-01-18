from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def assertBlockCountEqual(self, expected, got):
    self.assertEqual({hash(l) & 4294967295: c for l, c in expected.items()}, {h & 4294967295: c for h, c in got.items()})