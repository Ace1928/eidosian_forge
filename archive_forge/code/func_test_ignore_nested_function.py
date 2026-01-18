import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_ignore_nested_function(self):
    """Doctest module does not process doctest in nested functions."""
    self.flakes("\n        def doctest_stuff():\n            def inner_function():\n                '''\n                    >>> syntax error\n                    >>> inner_function()\n                    1\n                    >>> m\n                '''\n                return 1\n            m = inner_function()\n            return m\n        ")