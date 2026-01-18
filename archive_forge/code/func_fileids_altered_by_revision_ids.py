import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def fileids_altered_by_revision_ids(self, revision_ids):
    """This is a wrapper to strip TREE_ROOT if it occurs"""
    repo = self.branch.repository
    root_id = self.branch.basis_tree().path2id('')
    result = repo.fileids_altered_by_revision_ids(revision_ids)
    if root_id in result:
        del result[root_id]
    return result