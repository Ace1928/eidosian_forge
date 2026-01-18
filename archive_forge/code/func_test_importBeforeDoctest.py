import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_importBeforeDoctest(self):
    self.flakes("\n        import foo\n\n        def doctest_stuff():\n            '''\n                >>> foo\n            '''\n        ")