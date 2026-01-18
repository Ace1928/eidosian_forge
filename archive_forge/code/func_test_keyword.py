from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
def test_keyword(self):
    args = (1,)
    kwargs = dict(y=2, callback='old', z=3)
    self.assertEqual(self.replacer.get_old_value(args, kwargs), 'old')
    self.assertEqual(self.replacer.replace('new', args, kwargs), ('old', (1,), dict(y=2, callback='new', z=3)))