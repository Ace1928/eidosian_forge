from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def test_merge_text_returns_success(self):
    """A successful merge returns ('success', lines)."""
    changelog_merger, merge_hook_params = self.make_changelog_merger(b'', b'this text\n', b'other text\n')
    status, lines = changelog_merger.merge_contents(merge_hook_params)
    self.assertEqual(('success', [b'other text\n', b'this text\n']), (status, list(lines)))