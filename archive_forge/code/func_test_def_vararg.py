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
def test_def_vararg(self):
    suite = Suite('\ndef mysum(*others):\n    rv = 0\n    for n in others:\n        rv = rv + n\n    return rv\nx = mysum(1, 2, 3)\n')
    data = {}
    suite.execute(data)
    self.assertEqual(6, data['x'])