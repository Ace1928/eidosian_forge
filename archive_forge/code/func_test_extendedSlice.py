from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_extendedSlice(self):
    """
        Extended slices are supported.
        """
    self.flakes('\n        x = 3\n        [1, 2][x,:]\n        ')