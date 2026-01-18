import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_comment_directive(self):
    template = Template(utf8('{% comment blah blah %}foo'))
    self.assertEqual(template.generate(), b'foo')