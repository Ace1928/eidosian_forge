from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def test_merge_text_returns_not_applicable(self):
    """A conflict this plugin cannot resolve returns (not_applicable, None).
        """

    def entries_as_str(entries):
        return b''.join((entry + b'\n' for entry in entries))
    changelog_merger, merge_hook_params = self.make_changelog_merger(entries_as_str(sample2_base_entries), b'', entries_as_str(sample2_other_entries))
    self.assertEqual(('not_applicable', None), changelog_merger.merge_contents(merge_hook_params))