from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_command_with_backquotes(self):
    story = '\n$ foo = `brz file-id toto`\n'
    self.assertEqual([(['foo', '=', '`brz file-id toto`'], None, None, None)], script._script_to_commands(story))