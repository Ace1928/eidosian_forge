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
def test_binop_add(self):
    self.assertEqual(3, Expression('2 + 1').evaluate({}))
    self.assertEqual(3, Expression('x + y').evaluate({'x': 2, 'y': 1}))