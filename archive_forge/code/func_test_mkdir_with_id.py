from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import SettingFileIdUnsupported
def test_mkdir_with_id(self):
    t = self.make_branch_and_tree('t1')
    t.lock_write()
    self.addCleanup(t.unlock)
    if not t.supports_setting_file_ids():
        self.assertRaises((SettingFileIdUnsupported, TypeError), t.mkdir, 'path', b'my-id')
    else:
        file_id = t.mkdir('path', b'my-id')
        self.assertEqual(b'my-id', file_id)
        self.assertEqual('directory', t.kind('path'))