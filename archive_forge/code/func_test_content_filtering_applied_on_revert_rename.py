import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_content_filtering_applied_on_revert_rename(self):
    source, txt_path, bin_path = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
    if not source.supports_content_filtering():
        return
    self.assertFileEqual(b'Foo Txt', 'source/file1.txt')
    self.assert_basis_content(b'FOO TXT', source, txt_path)
    self.build_tree_contents([('source/file1.txt', b'Foo Txt with new content')])
    source.rename_one('file1.txt', 'file1.bin')
    self.assertTrue(os.path.exists('source/file1.bin'))
    self.assertFalse(os.path.exists('source/file1.txt'))
    self.assertFileEqual(b'Foo Txt with new content', 'source/file1.bin')
    source.revert(['file1.bin'])
    self.assertFalse(os.path.exists('source/file1.bin'))
    self.assertTrue(os.path.exists('source/file1.txt'))
    self.assertFileEqual(b'foo txt', 'source/file1.txt')