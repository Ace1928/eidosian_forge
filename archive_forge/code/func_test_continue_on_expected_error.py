from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_continue_on_expected_error(self):
    story = '\n$ brz not-a-command\n2>..."not-a-command"\n'
    self.run_script(story)