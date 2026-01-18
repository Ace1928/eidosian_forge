import datetime
from io import StringIO
import os
import sys
from unittest import mock
import unittest
from tornado.options import OptionParser, Error
from tornado.util import basestring_type
from tornado.test.util import subTest
import typing
def test_multiple_int(self):
    options = OptionParser()
    options.define('foo', type=int, multiple=True)
    options.parse_command_line(['main.py', '--foo=1,3,5:7'])
    self.assertEqual(options.foo, [1, 3, 5, 6, 7])