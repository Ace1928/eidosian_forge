from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_futureImportFirst(self):
    """
        __future__ imports must come before anything else.
        """
    self.flakes('\n        x = 5\n        from __future__ import division\n        ', m.LateFutureImport)
    self.flakes('\n        from foo import bar\n        from __future__ import division\n        bar\n        ', m.LateFutureImport)