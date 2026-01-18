import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_multi_includes(self):
    loader = DictLoader({'a.html': "{% include 'b.html' %}", 'b.html': "{% include 'c.html' %}", 'c.html': '{{1/0}}'})
    try:
        loader.load('a.html').generate()
        self.fail('did not get expected exception')
    except ZeroDivisionError:
        self.assertTrue('# c.html:1 (via b.html:1, a.html:1)' in traceback.format_exc())