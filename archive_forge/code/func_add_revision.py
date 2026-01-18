from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def add_revision(self, repo, revision_id, inv, parent_ids):
    """Add a revision with a given inventory and parents to a repository.

        :param repo: a repository.
        :param revision_id: the revision ID for the new revision.
        :param inv: an inventory (such as created by
            `make_one_file_inventory`).
        :param parent_ids: the parents for the new revision.
        """
    inv.revision_id = revision_id
    inv.root.revision = revision_id
    if repo.supports_rich_root():
        root_id = inv.root.file_id
        repo.texts.add_lines((root_id, revision_id), [], [])
    repo.add_inventory(revision_id, inv, parent_ids)
    revision = Revision(revision_id, committer='jrandom@example.com', timestamp=0, inventory_sha1='', timezone=0, message='foo', parent_ids=parent_ids)
    repo.add_revision(revision_id, revision, inv)