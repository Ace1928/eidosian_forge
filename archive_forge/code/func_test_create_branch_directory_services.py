import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_create_branch_directory_services(self):
    branch = self.make_branch('branch')
    tree = branch.create_checkout('tree', lightweight=True)

    class FooLookup:

        def look_up(self, name, url, purpose=None):
            return 'foo-' + name
    directories.register('foo:', FooLookup, 'Create branches named foo-')
    self.addCleanup(directories.remove, 'foo:')
    self.run_bzr('switch -b foo:branch2', working_dir='tree')
    tree = WorkingTree.open('tree')
    self.assertEndsWith(tree.branch.base, 'foo-branch2/')