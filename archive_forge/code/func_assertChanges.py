import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def assertChanges(self, branch, revno, expected_added=[], expected_removed=[], expected_modified=[], expected_renamed=[], expected_kind_changed=[]):
    """Check the changes introduced in a revision of a branch.

        This method checks that a revision introduces expected changes.
        The required changes are passed in as a list, where
        each entry contains the needed information about the change.

        If you do not wish to assert anything about a particular
        category then pass None instead.

        branch: The branch.
        revno: revision number of revision to check.
        expected_added: a list of (filename,) tuples that must have
            been added in the delta.
        expected_removed: a list of (filename,) tuples that must have
            been removed in the delta.
        expected_modified: a list of (filename,) tuples that must have
            been modified in the delta.
        expected_renamed: a list of (old_path, new_path) tuples that
            must have been renamed in the delta.
        expected_kind_changed: a list of (path, old_kind, new_kind) tuples
            that must have been changed in the delta.
        :return: revtree1, revtree2
        """
    repo = branch.repository
    revtree1 = repo.revision_tree(branch.get_rev_id(revno - 1))
    revtree2 = repo.revision_tree(branch.get_rev_id(revno))
    changes = revtree2.changes_from(revtree1)
    self._check_changes(changes, expected_added, expected_removed, expected_modified, expected_renamed, expected_kind_changed)
    return (revtree1, revtree2)