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
def test_access_undefined(self):
    expr = Expression('nothing', filename='index.html', lineno=50, lookup='lenient')
    retval = expr.evaluate({})
    assert isinstance(retval, Undefined)
    self.assertEqual('nothing', retval._name)
    assert retval._owner is UNDEFINED