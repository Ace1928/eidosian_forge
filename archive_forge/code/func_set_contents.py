import os
from breezy.branch import Branch
from breezy.osutils import pathjoin
from breezy.tests import TestCaseInTempDir, TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def set_contents(contents):
    self.build_tree_contents([('a/one', contents), ('b/two', contents), ('top', contents)])