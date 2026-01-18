import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_inaccessible_scope_class(self):
    """Doctest may not access class scope."""
    self.flakes("\n        class C:\n            def doctest_stuff(self):\n                '''\n                    >>> m\n                '''\n                return 1\n            m = 1\n        ", m.UndefinedName)