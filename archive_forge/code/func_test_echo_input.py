from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_echo_input(self):
    self.assertRaises(SyntaxError, self.run_script, '\n            $ echo <foo\n            ')