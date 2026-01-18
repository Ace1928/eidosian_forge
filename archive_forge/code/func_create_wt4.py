import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def create_wt4(self):
    control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
    control.create_repository()
    control.create_branch()
    tree = workingtree_4.WorkingTreeFormat4().initialize(control)
    return tree