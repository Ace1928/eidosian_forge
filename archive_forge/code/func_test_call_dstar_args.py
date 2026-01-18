import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
def test_call_dstar_args(self):

    def foo(x):
        return x
    expr = Expression('foo(**bar)')
    self.assertEqual(42, expr.evaluate({'foo': foo, 'bar': {'x': 42}}))