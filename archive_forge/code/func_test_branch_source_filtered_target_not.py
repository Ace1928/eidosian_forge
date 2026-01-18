import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_branch_source_filtered_target_not(self):
    source, txt_path, bin_path = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
    if not source.supports_content_filtering():
        return
    self.assertFileEqual(b'Foo Txt', 'source/file1.txt')
    self.assert_basis_content(b'FOO TXT', source, txt_path)
    self.run_bzr('branch source target')
    target = WorkingTree.open('target')
    self.assertFileEqual(b'FOO TXT', 'target/file1.txt')
    changes = target.changes_from(source.basis_tree())
    self.assertFalse(changes.has_changed())