import os
from ... import tests
from ..features import HardlinkFeature
def hardlinked(self):
    parent_stat = os.lstat(self.parent_tree.abspath('foo'))
    child_stat = os.lstat(self.child_tree.abspath('foo'))
    return parent_stat.st_ino == child_stat.st_ino