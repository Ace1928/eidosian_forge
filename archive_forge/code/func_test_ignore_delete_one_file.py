import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_ignore_delete_one_file(self):
    self.make_branch_and_working_tree()
    self.add_file('hello', b'foo')
    self.do_full_upload()
    self.add_file('.bzrignore-upload', b'hello')
    self.delete_any('hello')
    self.assertUpFileEqual(b'foo', 'hello')
    self.do_upload()
    self.assertUpFileEqual(b'foo', 'hello')