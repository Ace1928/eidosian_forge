from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_futureImportUsed(self):
    """__future__ is special, but names are injected in the namespace."""
    self.flakes('\n        from __future__ import division\n        from __future__ import print_function\n\n        assert print_function is not division\n        ')