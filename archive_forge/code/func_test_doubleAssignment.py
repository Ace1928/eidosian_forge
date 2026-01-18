from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
@skip('todo: Too hard to make this warn but other cases stay silent')
def test_doubleAssignment(self):
    """
        If a variable is re-assigned to without being used, no warning is
        emitted.
        """
    self.flakes('\n        x = 10\n        x = 20\n        ', m.RedefinedWhileUnused)