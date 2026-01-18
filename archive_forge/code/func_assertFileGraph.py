import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def assertFileGraph(self, expected_graph, tree, tip):
    tree.lock_read()
    self.addCleanup(tree.unlock)
    g = dict(tree.branch.repository.get_file_graph().iter_ancestry([tip]))
    self.assertEqual(expected_graph, g)