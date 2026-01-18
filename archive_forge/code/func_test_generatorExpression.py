from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_generatorExpression(self):
    """
        Don't warn when a variable in a generator expression is
        assigned to but not used.
        """
    self.flakes('\n        def f():\n            (None for i in range(10))\n        ')