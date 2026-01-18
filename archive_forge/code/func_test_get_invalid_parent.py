import sys
import breezy.errors
from breezy import urlutils
from breezy.osutils import getcwd
from breezy.tests import TestCaseWithTransport, TestNotApplicable, TestSkipped
def test_get_invalid_parent(self):
    b = self.make_branch('.')
    cwd = getcwd()
    n_dirs = len(cwd.split('/'))
    path = '../' * (n_dirs + 5) + 'foo'
    b.lock_write()
    b._set_parent_location(path)
    b.unlock()
    self.assertRaises(breezy.errors.InaccessibleParent, b.get_parent)