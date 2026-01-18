import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_delete_one_file_rename_to_deleted(self):
    self.make_branch_and_working_tree()
    self.add_file('a', b'foo')
    self.add_file('b', b'bar')
    self.do_full_upload()
    self.delete_any('a')
    self.rename_any('b', 'a')
    self.assertUpFileEqual(b'foo', 'a')
    self.do_upload()
    self.assertUpPathDoesNotExist('b')
    self.assertUpFileEqual(b'bar', 'a')