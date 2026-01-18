from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def test_acceptance_bug_723968(self):
    """Merging a branch that:

         1. adds a new entry, and
         2. edits an old entry (e.g. to fix a typo or twiddle formatting)

        will:

         1. add the new entry to the top
         2. keep the edit, without duplicating the edited entry or moving it.
        """
    result_entries = changelog_merge.merge_entries(sample_base_entries, sample_this_entries, sample_other_entries)
    self.assertEqual([b'Other entry O1', b'This entry T1', b'This entry T2', b'Base entry B1', b'Base entry B2 updated', b'Base entry B3'], list(result_entries))