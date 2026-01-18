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
def test_getattr_exception(self):

    class Something(object):

        def prop_a(self):
            raise NotImplementedError
        prop_a = property(prop_a)

        def prop_b(self):
            raise AttributeError
        prop_b = property(prop_b)
    self.assertRaises(NotImplementedError, Expression('s.prop_a').evaluate, {'s': Something()})
    self.assertRaises(AttributeError, Expression('s.prop_b').evaluate, {'s': Something()})