import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_non_ascii_name(self):
    loader = DictLoader({'tést.html': 'hello'})
    self.assertEqual(loader.load('tést.html').generate(), b'hello')