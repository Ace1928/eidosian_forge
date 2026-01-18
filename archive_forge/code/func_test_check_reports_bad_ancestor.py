from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_check_reports_bad_ancestor(self):
    repo = self.make_repo_with_extra_ghost_index()
    check_object = repo.check(['ignored'])
    check_object.report_results(verbose=False)
    self.assertContainsRe(self.get_log(), '1 revisions have incorrect parents in the revision index')
    check_object.report_results(verbose=True)
    self.assertContainsRe(self.get_log(), 'revision-id has wrong parents in index: \\(incorrect-parent\\) should be \\(\\)')