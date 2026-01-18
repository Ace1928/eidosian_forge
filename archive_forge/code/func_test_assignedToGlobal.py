from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_assignedToGlobal(self):
    """
        Binding an import to a declared global should not cause it to be
        reported as unused.
        """
    self.flakes('\n            def f(): global foo; import foo\n            def g(): foo.is_used()\n        ')