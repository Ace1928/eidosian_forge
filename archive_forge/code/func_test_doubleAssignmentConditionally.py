from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_doubleAssignmentConditionally(self):
    """
        If a variable is re-assigned within a conditional, no warning is
        emitted.
        """
    self.flakes('\n        x = 10\n        if True:\n            x = 20\n        ')