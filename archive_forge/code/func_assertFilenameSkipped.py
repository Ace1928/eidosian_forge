import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def assertFilenameSkipped(self, filename):
    tree = self.make_branch_and_tree('tree')
    try:
        self.build_tree(['tree/' + filename])
    except transport.NoSuchFile:
        if sys.platform == 'win32':
            raise tests.TestNotApplicable('Cannot create files named %r on win32' % (filename,))
    tree.smart_add(['tree'])
    self.assertFalse(tree.is_versioned(filename))