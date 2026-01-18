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
def test_augmented_attribute_assignment(self):
    suite = Suite("d['k'] += 42")
    d = {'k': 1}
    suite.execute({'d': d})
    self.assertEqual(43, d['k'])