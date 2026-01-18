import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_magicQualnameInClassScope(self):
    """
        Use of the C{__qualname__} magic builtin should not emit an undefined
        name warning if used in class scope.
        """
    self.flakes('__qualname__', m.UndefinedName)
    self.flakes('\n        class Foo:\n            __qualname__\n        ')
    self.flakes('\n        class Foo:\n            def bar(self):\n                __qualname__\n        ', m.UndefinedName)