import os
import stat
import sys
from breezy import tests
from breezy.bzr.branch import BzrBranch
from breezy.bzr.remote import RemoteBranchFormat
from breezy.controldir import ControlDir
from breezy.tests.test_permissions import check_mode_r
def test_mode_0(self):
    """Test when a transport returns null permissions for .bzr"""
    if isinstance(self.branch_format, RemoteBranchFormat):
        raise tests.TestNotApplicable('Remote branches have no permission logic')
    self.make_branch_and_tree('.')
    bzrdir = ControlDir.open('.')
    _orig_stat = bzrdir.transport.stat

    def null_perms_stat(*args, **kwargs):
        result = _orig_stat(*args, **kwargs)
        return _NullPermsStat(result)
    bzrdir.transport.stat = null_perms_stat
    self.assertIs(None, bzrdir._get_dir_mode())
    self.assertIs(None, bzrdir._get_file_mode())