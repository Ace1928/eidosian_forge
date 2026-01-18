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
def test_assign(self):
    suite = Suite('foo = 42')
    data = {}
    suite.execute(data)
    self.assertEqual(42, data['foo'])