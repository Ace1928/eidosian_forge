import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_nested_doctest_ignored(self):
    """Check that nested doctests are ignored."""
    checker = self.flakes('\n        m = None\n\n        def doctest_stuff():\n            \'\'\'\n                >>> def function_in_doctest():\n                ...     """\n                ...     >>> ignored_undefined_name\n                ...     """\n                ...     df = m\n                ...     return df\n                ...\n                >>> function_in_doctest()\n            \'\'\'\n            f = m\n            return f\n        ')
    scopes = checker.deadScopes
    module_scopes = [scope for scope in scopes if scope.__class__ is ModuleScope]
    doctest_scopes = [scope for scope in scopes if scope.__class__ is DoctestScope]
    function_scopes = [scope for scope in scopes if scope.__class__ is FunctionScope]
    self.assertEqual(len(module_scopes), 1)
    self.assertEqual(len(doctest_scopes), 1)
    module_scope = module_scopes[0]
    doctest_scope = doctest_scopes[0]
    self.assertIn('m', module_scope)
    self.assertIn('doctest_stuff', module_scope)
    self.assertIn('function_in_doctest', doctest_scope)
    self.assertEqual(len(function_scopes), 2)
    self.assertIn('f', function_scopes[0])
    self.assertIn('df', function_scopes[1])