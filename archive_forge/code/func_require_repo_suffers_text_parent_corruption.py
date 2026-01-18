from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def require_repo_suffers_text_parent_corruption(self, repo):
    if not repo._reconcile_fixes_text_parents:
        raise TestNotApplicable('Format does not support text parent reconciliation')