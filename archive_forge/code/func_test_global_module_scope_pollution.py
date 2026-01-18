import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_global_module_scope_pollution(self):
    """Check that global in doctest does not pollute module scope."""
    checker = self.flakes("\n        def doctest_stuff():\n            '''\n                >>> def function_in_doctest():\n                ...     global m\n                ...     m = 50\n                ...     df = 10\n                ...     m = df\n                ...\n                >>> function_in_doctest()\n            '''\n            f = 10\n            return f\n\n        ")
    scopes = checker.deadScopes
    module_scopes = [scope for scope in scopes if scope.__class__ is ModuleScope]
    doctest_scopes = [scope for scope in scopes if scope.__class__ is DoctestScope]
    function_scopes = [scope for scope in scopes if scope.__class__ is FunctionScope]
    self.assertEqual(len(module_scopes), 1)
    self.assertEqual(len(doctest_scopes), 1)
    module_scope = module_scopes[0]
    doctest_scope = doctest_scopes[0]
    self.assertIn('doctest_stuff', module_scope)
    self.assertIn('function_in_doctest', doctest_scope)
    self.assertEqual(len(function_scopes), 2)
    self.assertIn('f', function_scopes[0])
    self.assertIn('df', function_scopes[1])
    self.assertIn('m', function_scopes[1])
    self.assertNotIn('m', module_scope)