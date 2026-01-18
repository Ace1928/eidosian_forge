from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_futureImportUndefined(self):
    """Importing undefined names from __future__ fails."""
    self.flakes('\n        from __future__ import print_statement\n        ', m.FutureFeatureNotDefined)