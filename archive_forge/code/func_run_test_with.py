import copy
import functools
import itertools
import sys
import types
import unittest
import warnings
from testtools.compat import reraise
from testtools import content
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.matchers._basic import _FlippedEquals
from testtools.monkey import patch
from testtools.runtest import (
from testtools.testresult import (
def run_test_with(test_runner, **kwargs):
    """Decorate a test as using a specific ``RunTest``.

    e.g.::

      @run_test_with(CustomRunner, timeout=42)
      def test_foo(self):
          self.assertTrue(True)

    The returned decorator works by setting an attribute on the decorated
    function.  `TestCase.__init__` looks for this attribute when deciding on a
    ``RunTest`` factory.  If you wish to use multiple decorators on a test
    method, then you must either make this one the top-most decorator, or you
    must write your decorators so that they update the wrapping function with
    the attributes of the wrapped function.  The latter is recommended style
    anyway.  ``functools.wraps``, ``functools.wrapper`` and
    ``twisted.python.util.mergeFunctionMetadata`` can help you do this.

    :param test_runner: A ``RunTest`` factory that takes a test case and an
        optional list of exception handlers.  See ``RunTest``.
    :param kwargs: Keyword arguments to pass on as extra arguments to
        'test_runner'.
    :return: A decorator to be used for marking a test as needing a special
        runner.
    """

    def decorator(function):

        def _run_test_with(case, handlers=None, last_resort=None):
            try:
                return test_runner(case, handlers=handlers, last_resort=last_resort, **kwargs)
            except TypeError:
                return test_runner(case, handlers=handlers, **kwargs)
        function._run_test_with = _run_test_with
        return function
    return decorator