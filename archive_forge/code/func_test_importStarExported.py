from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importStarExported(self):
    """
        Report undefined if import * is used
        """
    self.flakes("\n        from math import *\n        __all__ = ['sin', 'cos']\n        csc(1)\n        ", m.ImportStarUsed, m.ImportStarUsage, m.ImportStarUsage, m.ImportStarUsage)