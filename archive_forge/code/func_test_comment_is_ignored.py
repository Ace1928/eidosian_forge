from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_comment_is_ignored(self):
    self.assertEqual([], script._script_to_commands('#comment\n'))