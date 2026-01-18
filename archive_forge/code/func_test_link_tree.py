import os
from ... import tests
from ..features import HardlinkFeature
def test_link_tree(self):
    """Ensure the command works as intended"""
    os.chdir('child')
    self.parent_tree.unlock()
    self.run_bzr('link-tree ../parent')
    self.assertTrue(self.hardlinked())
    self.parent_tree.lock_write()