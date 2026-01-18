from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assignToGlobal(self):
    """
        Assigning to a global and then not using that global is perfectly
        acceptable. Do not mistake it for an unused local variable.
        """
    self.flakes('\n        b = 0\n        def a():\n            global b\n            b = 1\n        ')