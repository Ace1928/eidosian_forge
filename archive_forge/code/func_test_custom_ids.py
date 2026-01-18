import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_custom_ids(self):
    sio = StringIO()
    action = test_smart_add.AddCustomIDAction(to_file=sio, should_print=True)
    self.build_tree(['file1', 'dir1/', 'dir1/file2'])
    wt = self.make_branch_and_tree('.')
    if not wt._format.supports_setting_file_ids:
        self.assertRaises(workingtree.SettingFileIdUnsupported, wt.smart_add, ['.'], action=action)
        return
    wt.smart_add(['.'], action=action)
    sio.seek(0)
    lines = sorted(sio.readlines())
    self.assertEqual(['added dir1 with id directory-dir1\n', 'added dir1/file2 with id file-dir1%file2\n', 'added file1 with id file-file1\n'], lines)
    wt.lock_read()
    self.addCleanup(wt.unlock)
    self.assertEqual([('', wt.path2id('')), ('dir1', b'directory-dir1'), ('file1', b'file-file1'), ('dir1/file2', b'file-dir1%file2')], [(path, ie.file_id) for path, ie in wt.iter_entries_by_dir()])