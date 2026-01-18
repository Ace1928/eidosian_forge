import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def make_reference(self, name):
    tree = self.make_branch_and_tree(name)
    if not tree.branch.repository._format.rich_root_data:
        raise tests.TestNotApplicable('format does not support rich roots')
    tree.commit('foo')
    return tree