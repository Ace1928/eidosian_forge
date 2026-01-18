from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
def test__check_for_inconsistent_revision_parents(self):
    """_check_for_inconsistent_revision_parents raises BzrCheckError if
        there are any revisions with inconsistent parents.
        """
    repo = self.make_repo_with_extra_ghost_index()
    self.assertRaises(errors.BzrCheckError, repo._check_for_inconsistent_revision_parents)