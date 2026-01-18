import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_full_upload_empty_tree(self):
    self.make_branch_and_working_tree()
    self.do_full_upload()
    revid_path = self.tree.branch.get_config_stack().get('upload_revid_location')
    self.assertUpPathExists(revid_path)