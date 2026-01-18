from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
def test__check_for_inconsistent_revision_parents_on_clean_repo(self):
    """_check_for_inconsistent_revision_parents does nothing if there are
        no broken revisions.
        """
    repo = self.make_repository('empty-repo')
    if not repo._format.revision_graph_can_have_wrong_parents:
        raise TestNotApplicable('%r cannot have corrupt revision index.' % repo)
    with repo.lock_read():
        repo._check_for_inconsistent_revision_parents()