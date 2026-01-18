import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_fileids_involved_full_compare(self):
    self.branch.lock_read()
    self.addCleanup(self.branch.unlock)
    pp = []
    graph = self.branch.repository.get_graph()
    history = list(graph.iter_lefthand_ancestry(self.branch.last_revision(), [_mod_revision.NULL_REVISION]))
    history.reverse()
    if len(history) < 2:
        return
    for start in range(0, len(history) - 1):
        start_id = history[start]
        for end in range(start + 1, len(history)):
            end_id = history[end]
            unique_revs = graph.find_unique_ancestors(end_id, [start_id])
            l1 = self.branch.repository.fileids_altered_by_revision_ids(unique_revs)
            l1 = set(l1.keys())
            l2 = self.compare_tree_fileids(self.branch, start_id, end_id)
            self.assertEqual(l1, l2)