from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_moduleWithYield(self):
    """
        If a yield is used at the module level, a warning is emitted.
        """
    self.flakes('\n        yield\n        ', m.YieldOutsideFunction)