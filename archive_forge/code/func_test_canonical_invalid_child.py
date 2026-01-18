from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_canonical_invalid_child(self):
    work_tree = self._make_canonical_test_tree()
    if features.CaseInsensitiveFilesystemFeature.available():
        self.assertEqual('dir/None', work_tree.get_canonical_path('Dir/None'))
    elif features.CaseInsCasePresFilenameFeature.available():
        self.assertEqual('dir/None', work_tree.get_canonical_path('Dir/None'))
    else:
        self.assertEqual('Dir/None', work_tree.get_canonical_path('Dir/None'))