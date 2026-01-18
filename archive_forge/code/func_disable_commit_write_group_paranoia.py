import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def disable_commit_write_group_paranoia(self, repo):
    if isinstance(repo, remote.RemoteRepository):
        repo.abort_write_group()
        raise tests.TestSkipped('repository format does not support storing revisions with missing texts.')
    pack_coll = getattr(repo, '_pack_collection', None)
    if pack_coll is not None:
        pack_coll._check_new_inventories = lambda: []