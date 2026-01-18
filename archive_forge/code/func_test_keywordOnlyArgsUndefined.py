import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_keywordOnlyArgsUndefined(self):
    """Typo in kwonly name."""
    self.flakes('\n        def f(*, a, b=default_c):\n            print(a, b)\n        ', m.UndefinedName)