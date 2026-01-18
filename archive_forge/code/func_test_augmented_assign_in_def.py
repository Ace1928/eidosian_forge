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
def test_augmented_assign_in_def(self):
    d = {}
    Suite('def foo():\n    i = 1\n    i += 1\n    return i\nx = foo()').execute(d)
    self.assertEqual(2, d['x'])