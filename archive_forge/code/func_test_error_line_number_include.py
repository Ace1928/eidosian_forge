import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_error_line_number_include(self):
    loader = DictLoader({'base.html': "{% include 'sub.html' %}", 'sub.html': '{{1/0}}'})
    try:
        loader.load('base.html').generate()
        self.fail('did not get expected exception')
    except ZeroDivisionError:
        self.assertTrue('# sub.html:1 (via base.html:1)' in traceback.format_exc())