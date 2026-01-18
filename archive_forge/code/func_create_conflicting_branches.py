import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def create_conflicting_branches(self):
    """Create two branches which have overlapping modifications.

        :return: (tree, other_branch) Where merging other_branch causes a file
            conflict.
        """
    builder = self.make_branch_builder('branch')
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('fname', b'f-id', 'file', b'a\nb\nc\n'))], revision_id=b'rev1')
    builder.build_snapshot([b'rev1'], [('modify', ('fname', b'a\nB\nD\n'))], revision_id=b'rev2other')
    other = builder.get_branch().controldir.sprout('other').open_branch()
    builder.build_snapshot([b'rev1'], [('modify', ('fname', b'a\nB\nC\n'))], revision_id=b'rev2this')
    tree = builder.get_branch().create_checkout('tree', lightweight=True)
    return (tree, other)