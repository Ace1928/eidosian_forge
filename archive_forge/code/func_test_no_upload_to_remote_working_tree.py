import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_no_upload_to_remote_working_tree(self):
    cmd = self._get_cmd_upload()
    up_url = self.get_url(self.branch_dir)
    self.assertRaises(cmds.CannotUploadToWorkingTree, cmd.run, up_url, directory=self.remote_branch_url)