from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
def test__find_inconsistent_revision_parents(self):
    """_find_inconsistent_revision_parents finds revisions with broken
        parents.
        """
    repo = self.make_repo_with_extra_ghost_index()
    self.assertEqual([(b'revision-id', (b'incorrect-parent',), ())], list(repo._find_inconsistent_revision_parents()))