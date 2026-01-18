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
def test_def_kwonlyarg_with_default(self):
    suite = Suite('\ndef kwonly(*args, k="bar"):\n    return k\nx = kwonly(k="foo")\ny = kwonly()\n')
    data = {}
    suite.execute(data)
    self.assertEqual('foo', data['x'])
    self.assertEqual('bar', data['y'])