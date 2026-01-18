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
def test_name_lookup(self):
    self.assertEqual('bar', Expression('foo').evaluate({'foo': 'bar'}))
    self.assertEqual(id, Expression('id').evaluate({}))
    self.assertEqual('bar', Expression('id').evaluate({'id': 'bar'}))
    self.assertEqual(None, Expression('id').evaluate({'id': None}))