import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_indentationErrorInDoctest(self):
    exc = self.flakes('\n        def doctest_stuff():\n            """\n                >>> if True:\n                ... pass\n            """\n        ', m.DoctestSyntaxError).messages[0]
    self.assertEqual(exc.lineno, 5)
    self.assertEqual(exc.col, 13)