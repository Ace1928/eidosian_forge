from breezy import tests
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_canonical_tree_name_mismatch(self):
    self.requireFeature(features.case_sensitive_filesystem_feature)
    work_tree = self.make_branch_and_tree('.')
    self.build_tree(['test/', 'test/file', 'Test'])
    work_tree.add(['test/', 'test/file', 'Test'])
    self.assertEqual(['test', 'Test', 'test/file', 'Test/file'], list(work_tree.get_canonical_paths(['test', 'Test', 'test/file', 'Test/file'])))
    test_revid = work_tree.commit('commit')
    test_tree = work_tree.branch.repository.revision_tree(test_revid)
    test_tree.lock_read()
    self.addCleanup(test_tree.unlock)
    self.assertEqual(['', 'Test', 'test', 'test/file'], [p for p, e in test_tree.iter_entries_by_dir()])