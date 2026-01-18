from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_command_with_double_quoted_param(self):
    story = '$ brz commit -m "two words" '
    self.assertEqual([(['brz', 'commit', '-m', '"two words"'], None, None, None)], script._script_to_commands(story))