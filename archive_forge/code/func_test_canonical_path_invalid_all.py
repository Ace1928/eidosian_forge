from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_canonical_path_invalid_all(self):
    work_tree = self._make_canonical_test_tree()
    self.assertEqual('foo/bar', work_tree.get_canonical_path('foo/bar'))