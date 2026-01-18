from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def assertSimilar(self, expected_score, blob1, blob2):
    self.assertEqual(expected_score, _similarity_score(blob1, blob2))
    self.assertEqual(expected_score, _similarity_score(blob2, blob1))