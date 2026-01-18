from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_ignored_when_not_directly_assigned(self):
    self.flakes('\n        import bar\n        (__all__,) = ("foo",)\n        ', m.UnusedImport)