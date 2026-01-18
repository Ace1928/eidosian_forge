from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def apply_inventory_Inventory(self, basis, delta, invalid_delta=True):
    """Apply delta to basis and return the result.

    :param basis: An inventory to be used as the basis.
    :param delta: The inventory delta to apply:
    :return: An inventory resulting from the application.
    """
    basis.apply_delta(delta)
    return basis