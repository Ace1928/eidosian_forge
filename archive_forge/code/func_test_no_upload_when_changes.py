import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_no_upload_when_changes(self):
    self.make_branch_and_working_tree()
    self.add_file('a', b'foo')
    self.set_file_content('a', b'bar')
    self.assertRaises(errors.UncommittedChanges, self.do_upload)