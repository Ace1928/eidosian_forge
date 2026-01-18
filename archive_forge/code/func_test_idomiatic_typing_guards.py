from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_idomiatic_typing_guards(self):
    self.flakes('\n            from typing import TYPE_CHECKING\n\n            if TYPE_CHECKING:\n                from t import T\n\n            def f() -> T:\n                pass\n        ')
    self.flakes('\n            if False:\n                from t import T\n\n            def f() -> T:\n                pass\n        ')
    self.flakes('\n            MYPY = False\n\n            if MYPY:\n                from t import T\n\n            def f() -> T:\n                pass\n        ')