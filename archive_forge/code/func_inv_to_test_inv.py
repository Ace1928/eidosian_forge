from breezy import tests
from breezy.bzr import groupcompress
def inv_to_test_inv(self, inv):
    """Convert a regular Inventory object into an inventory under test."""
    return self._inv_to_test_inv(self, inv)