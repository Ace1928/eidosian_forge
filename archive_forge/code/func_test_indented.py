from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_indented(self):
    story = '\n            $ brz add\n            adding file\n            adding file2\n            '
    self.assertEqual([(['brz', 'add'], None, 'adding file\nadding file2\n', None)], script._script_to_commands(story))