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
def test_def_some_defaults(self):
    suite = Suite('\ndef difference(v1, v2=10):\n    return v1 - v2\nx = difference(20, 19)\ny = difference(20)\n')
    data = {}
    suite.execute(data)
    self.assertEqual(1, data['x'])
    self.assertEqual(10, data['y'])