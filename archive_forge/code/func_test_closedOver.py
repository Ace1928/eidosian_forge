from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_closedOver(self):
    """
        Don't warn when the assignment is used in an inner function.
        """
    self.flakes('\n        def barMaker():\n            foo = 5\n            def bar():\n                return foo\n            return bar\n        ')