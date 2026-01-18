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
def test_error_access_undefined(self):
    expr = Expression('nothing', filename='index.html', lineno=50, lookup='strict')
    try:
        expr.evaluate({})
        self.fail('Expected UndefinedError')
    except UndefinedError as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        frame = exc_traceback.tb_next
        frames = []
        while frame.tb_next:
            frame = frame.tb_next
            frames.append(frame)
        self.assertEqual('"nothing" not defined', str(e))
        self.assertEqual("<Expression 'nothing'>", frames[-3].tb_frame.f_code.co_name)
        self.assertEqual('index.html', frames[-3].tb_frame.f_code.co_filename)
        self.assertEqual(50, frames[-3].tb_lineno)