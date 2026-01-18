import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_custom_escape(self):
    loader = DictLoader({'foo.py': '{% autoescape py_escape %}s = {{ name }}\n'})

    def py_escape(s):
        self.assertEqual(type(s), bytes)
        return repr(native_str(s))

    def render(template, name):
        return loader.load(template).generate(py_escape=py_escape, name=name)
    self.assertEqual(render('foo.py', '<html>'), b"s = '<html>'\n")
    self.assertEqual(render('foo.py', "';sys.exit()"), b's = "\';sys.exit()"\n')
    self.assertEqual(render('foo.py', ['not a string']), b's = "[\'not a string\']"\n')