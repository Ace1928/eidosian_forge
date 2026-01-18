import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def make_remote_branch_without_working_tree(self):
    """Creates a branch without working tree to upload from.

        It's created from the existing self.branch_dir one which still has its
        working tree.
        """
    self.make_branch_and_working_tree()
    self.add_file('hello', b'foo')
    remote_branch_url = self.get_url(self.remote_branch_dir)
    self.run_bzr(['push', remote_branch_url, '--directory', self.branch_dir])
    return remote_branch_url