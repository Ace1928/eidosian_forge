from breezy import tests
from breezy.memorytree import MemoryTree
from breezy.tests import TestCaseWithTransport
from breezy.treebuilder import AlreadyBuilding, NotBuilding, TreeBuilder
def test_build_tree_not_started_errors(self):
    builder = TreeBuilder()
    self.assertRaises(NotBuilding, builder.build, 'foo')