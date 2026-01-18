from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_comment_multiple_lines(self):
    self.assertEqual([(['bar'], None, None, None)], script._script_to_commands('\n            # this comment is ignored\n            # so is this\n            # no we run bar\n            $ bar\n            '))