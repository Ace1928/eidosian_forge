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
def test_getitem_undefined_string(self):

    class Something(object):

        def __repr__(self):
            return '<Something>'
    something = Something()
    expr = Expression('something["nil"]', filename='index.html', lineno=50, lookup='lenient')
    retval = expr.evaluate({'something': something})
    assert isinstance(retval, Undefined)
    self.assertEqual('nil', retval._name)
    assert retval._owner is something