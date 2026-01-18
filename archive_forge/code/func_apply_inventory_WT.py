from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def apply_inventory_WT(self, basis, delta, invalid_delta=True):
    """Apply delta to basis and return the result.

    This sets the tree state to be basis, and then calls apply_inventory_delta.

    :param basis: An inventory to be used as the basis.
    :param delta: The inventory delta to apply:
    :return: An inventory resulting from the application.
    """
    control = self.make_controldir('tree', format=self.format._matchingcontroldir)
    control.create_repository()
    control.create_branch()
    tree = self.format.initialize(control)
    tree.lock_write()
    try:
        tree._write_inventory(basis)
    finally:
        tree.unlock()
    tree = tree.controldir.open_workingtree()
    tree.lock_write()
    try:
        tree.apply_inventory_delta(delta)
    finally:
        tree.unlock()
    tree = tree.controldir.open_workingtree()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    if not invalid_delta:
        tree._validate()
    return tree.root_inventory