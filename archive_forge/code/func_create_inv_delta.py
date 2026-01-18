import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_inv_delta(self, delta, rev_id):
    """Translate a 'delta shape' into an actual InventoryDelta"""
    dir_ids = {'': b'root-id'}
    inv_delta = []
    for old_path, new_path, file_id in delta:
        if old_path is not None and old_path.endswith('/'):
            old_path = old_path[:-1]
        if new_path is None:
            inv_delta.append((old_path, None, file_id, None))
            continue
        ie = self.path_to_ie(new_path, file_id, rev_id, dir_ids)
        inv_delta.append((old_path, new_path, file_id, ie))
    return inv_delta