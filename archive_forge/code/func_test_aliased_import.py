from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_aliased_import(self):
    """Detect when typing is imported as another name"""
    self.flakes('\n        import typing as t\n\n        @t.overload\n        def f(s: None) -> None:\n            pass\n\n        @t.overload\n        def f(s: int) -> int:\n            pass\n\n        def f(s):\n            return s\n        ')