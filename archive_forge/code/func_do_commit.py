import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def do_commit():
    try:
        list(builder.record_iter_changes(tree, tree.last_revision(), []))
        builder.finish_inventory()
    except:
        builder.abort()
        raise
    else:
        builder.commit('msg')