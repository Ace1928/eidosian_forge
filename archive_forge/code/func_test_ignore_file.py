import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_ignore_file(self):
    self.make_branch_and_working_tree()
    self.do_full_upload()
    self.add_file('.bzrignore-upload', b'foo')
    self.add_file('foo', b'bar')
    self.do_upload()
    self.assertUpPathDoesNotExist('foo')