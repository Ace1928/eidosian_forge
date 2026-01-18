from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_doubleAssignmentWithUse(self):
    """
        If a variable is re-assigned to after being used, no warning is
        emitted.
        """
    self.flakes('\n        x = 10\n        y = x * 2\n        x = 20\n        ')