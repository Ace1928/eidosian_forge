from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_futureImportStar(self):
    """Importing '*' from __future__ fails."""
    self.flakes('\n        from __future__ import *\n        ', m.FutureFeatureNotDefined)