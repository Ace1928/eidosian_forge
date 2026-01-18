import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_doctestCanReferToClass(self):
    self.flakes("\n        class Foo():\n            '''\n                >>> Foo\n            '''\n            def bar(self):\n                '''\n                    >>> Foo\n                '''\n        ")