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
def test_internal_shadowing(self):
    suite = Suite('data = []\nbar = foo\n')
    data = {'foo': 42}
    suite.execute(data)
    self.assertEqual(42, data['bar'])