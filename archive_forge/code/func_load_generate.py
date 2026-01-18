import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
def load_generate(path, **kwargs):
    assert loader is not None
    return loader.load(path).generate(**kwargs)