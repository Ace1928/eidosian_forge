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
def test_def_with_multiple_statements(self):
    suite = Suite('\ndef donothing():\n    if True:\n        return foo\n')
    data = {'foo': 'bar'}
    suite.execute(data)
    assert 'donothing' in data
    self.assertEqual('bar', data['donothing']())