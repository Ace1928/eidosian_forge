import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def test_break_in_apply(self):
    try:
        Template(utf8('{% for i in [] %}{% apply foo %}{% break %}{% end %}{% end %}'))
        raise Exception('Did not get expected exception')
    except ParseError:
        pass