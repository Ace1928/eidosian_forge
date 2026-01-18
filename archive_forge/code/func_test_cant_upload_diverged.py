import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_cant_upload_diverged(self):
    self.assertRaises(cmds.DivergedUploadedTree, self.do_incremental_upload, directory=self.diverged_tree.basedir)
    self.assertRevidUploaded(b'rev2a')