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
def test_exec(self):
    suite = Suite("x = 1; exec(d['k']); assert x == 42, x")
    suite.execute({'d': {'k': 'x = 42'}})