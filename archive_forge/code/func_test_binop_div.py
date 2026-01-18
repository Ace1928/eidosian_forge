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
def test_binop_div(self):
    self.assertEqual(2, Expression('4 / 2').evaluate({}))
    self.assertEqual(2, Expression('x / y').evaluate({'x': 4, 'y': 2}))