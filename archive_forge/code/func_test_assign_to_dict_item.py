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
def test_assign_to_dict_item(self):
    suite = Suite("d['k'] = 'foo'")
    data = {'d': {}}
    suite.execute(data)
    self.assertEqual('foo', data['d']['k'])