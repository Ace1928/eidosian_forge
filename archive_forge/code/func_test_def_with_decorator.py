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
def test_def_with_decorator(self):
    suite = Suite("\ndef lower(fun):\n    return lambda: fun().lower()\n\n@lower\ndef say_hi():\n    return 'Hi!'\n\nresult = say_hi()\n")
    data = {}
    suite.execute(data)
    self.assertEqual('hi!', data['result'])