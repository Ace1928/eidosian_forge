from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_echo_to_output(self):
    retcode, out, err = self.run_command(['echo'], None, '\n', None)
    self.assertEqual('\n', out)
    self.assertEqual(None, err)