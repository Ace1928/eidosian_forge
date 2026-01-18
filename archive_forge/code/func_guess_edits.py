from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def guess_edits(new, deleted):
    return changelog_merge.default_guess_edits(new, deleted, entry_as_str=lambda x: x)