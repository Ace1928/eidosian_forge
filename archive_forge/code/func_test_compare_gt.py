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
def test_compare_gt(self):
    self.assertEqual(True, Expression('2 > 1').evaluate({}))
    self.assertEqual(True, Expression('x > y').evaluate({'x': 2, 'y': 1}))