import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_create_remote_dir_twice(self):
    self.make_branch_and_working_tree()
    self.add_dir('dir')
    self.do_full_upload()
    self.add_file('dir/goodbye', b'baz')
    self.assertUpPathDoesNotExist('dir/goodbye')
    self.do_full_upload()
    self.assertUpFileEqual(b'baz', 'dir/goodbye')
    self.assertUpPathModeEqual('dir', 509)