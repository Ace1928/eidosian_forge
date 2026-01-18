from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_if_tuple(self):
    """
        Test C{if (foo,)} conditions.
        """
    self.flakes('if (): pass')
    self.flakes('\n        if (\n            True\n        ):\n            pass\n        ')
    self.flakes('\n        if (\n            True,\n        ):\n            pass\n        ', m.IfTuple)
    self.flakes('\n        x = 1 if (\n            True,\n        ) else 2\n        ', m.IfTuple)