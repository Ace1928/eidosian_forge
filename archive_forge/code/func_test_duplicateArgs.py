from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_duplicateArgs(self):
    self.flakes('def fu(bar, bar): pass', m.DuplicateArgument)