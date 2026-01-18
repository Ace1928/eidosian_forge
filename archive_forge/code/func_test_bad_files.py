import os
from io import StringIO
from .. import errors
from ..status import show_tree_status
from . import TestCaseWithTransport
from .features import OsFifoFeature
def test_bad_files(self):
    """Test that bzr will ignore files it doesn't like"""
    self.requireFeature(OsFifoFeature)
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    files = ['one', 'two', 'three']
    file_ids = [b'one-id', b'two-id', b'three-id']
    self.build_tree(files)
    wt.add(files, ids=file_ids)
    wt.commit('Commit one', rev_id=b'a@u-0-0')
    verify_status(self, wt, [])
    os.mkfifo('a-fifo')
    self.build_tree(['six'])
    verify_status(self, wt, ['unknown:\n', '  a-fifo\n', '  six\n'])
    self.assertRaises(errors.BadFileKindError, wt.smart_add, ['a-fifo'])
    verify_status(self, wt, ['unknown:\n', '  a-fifo\n', '  six\n'])
    wt.smart_add([])
    verify_status(self, wt, ['added:\n', '  six\n', 'unknown:\n', '  a-fifo\n'])
    wt.commit('Commit four', rev_id=b'a@u-0-3')
    verify_status(self, wt, ['unknown:\n', '  a-fifo\n'])