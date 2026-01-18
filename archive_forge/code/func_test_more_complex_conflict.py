from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def test_more_complex_conflict(self):
    """Like test_acceptance_bug_723968, but with a more difficult conflict:
        the new entry and the edited entry are adjacent.
        """

    def guess_edits(new, deleted):
        return changelog_merge.default_guess_edits(new, deleted, entry_as_str=lambda x: x)
    result_entries = changelog_merge.merge_entries(sample2_base_entries, sample2_this_entries, sample2_other_entries, guess_edits=guess_edits)
    self.assertEqual([b'Other entry O1', b'This entry T1', b'This entry T2', b'Base entry B1 edit', b'Base entry B2'], list(result_entries))