from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_all_with_names(self):
    self.flakes('\n        from foo import bar\n        __all__ = [bar]\n        ')