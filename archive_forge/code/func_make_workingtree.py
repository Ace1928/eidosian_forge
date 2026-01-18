import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def make_workingtree(self, relpath=''):
    url = self.get_url(relpath)
    if relpath:
        self.build_tree([relpath + '/'])
    dir = bzrdir.BzrDirMetaFormat1().initialize(url)
    dir.create_repository()
    dir.create_branch()
    try:
        return workingtree_4.WorkingTreeFormat4().initialize(dir)
    except errors.NotLocalUrl:
        raise TestSkipped('Not a local URL')