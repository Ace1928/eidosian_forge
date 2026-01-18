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
def test_array_indices(self):
    data = dict(items=[1, 2, 3])
    self.assertEqual(1, Expression('items[0]').evaluate(data))
    self.assertEqual(3, Expression('items[-1]').evaluate(data))