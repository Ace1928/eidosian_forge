import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_ignore_change_dir_into_file(self):
    self.make_branch_and_working_tree()
    self.add_dir('hello')
    self.add_file('hello/file', b'foo')
    self.do_full_upload()
    self.add_file('.bzrignore-upload', b'hello')
    self.delete_any('hello/file')
    self.transform_dir_into_file('hello', b'bar')
    self.assertUpFileEqual(b'foo', 'hello/file')
    self.do_upload()
    self.assertUpFileEqual(b'foo', 'hello/file')