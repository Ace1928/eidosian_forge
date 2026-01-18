import os
import stat
import sys
from breezy import tests
from breezy.bzr.branch import BzrBranch
from breezy.bzr.remote import RemoteBranchFormat
from breezy.controldir import ControlDir
from breezy.tests.test_permissions import check_mode_r
def null_perms_stat(*args, **kwargs):
    result = _orig_stat(*args, **kwargs)
    return _NullPermsStat(result)