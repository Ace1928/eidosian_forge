import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_tree_from_shape(self, rev_id, shape):
    dir_ids = {'': b'root-id'}
    inv = inventory.Inventory(b'root-id', rev_id)
    for info in shape:
        if len(info) == 2:
            path, file_id = info
            ie_rev_id = rev_id
        else:
            path, file_id, ie_rev_id = info
        if path == '':
            del inv._byid[inv.root.file_id]
            inv.root.file_id = file_id
            inv._byid[file_id] = inv.root
            dir_ids[''] = file_id
            continue
        inv.add(self.path_to_ie(path, file_id, ie_rev_id, dir_ids))
    return inventorytree.InventoryRevisionTree(_Repo(), inv, rev_id)