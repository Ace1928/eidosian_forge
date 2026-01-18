import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_manual_minimize_whitespace(self):
    loader = DictLoader({'foo.txt': '{% for i in items\n  %}{% if i > 0 %}, {% end %}{#\n  #}{{i\n  }}{% end\n%}'})
    self.assertEqual(loader.load('foo.txt').generate(items=range(5)), b'0, 1, 2, 3, 4')