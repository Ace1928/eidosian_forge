import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def setup_module_clash(self):

    class Module(object):
        __file__ = 'bar/foo.py'
    sys.modules['foo'] = Module
    full_path = os.path.abspath('foo')
    original_listdir = os.listdir
    original_isfile = os.path.isfile
    original_isdir = os.path.isdir
    original_realpath = os.path.realpath

    def cleanup():
        os.listdir = original_listdir
        os.path.isfile = original_isfile
        os.path.isdir = original_isdir
        os.path.realpath = original_realpath
        del sys.modules['foo']
        if full_path in sys.path:
            sys.path.remove(full_path)
    self.addCleanup(cleanup)

    def listdir(_):
        return ['foo.py']

    def isfile(_):
        return True

    def isdir(_):
        return True
    os.listdir = listdir
    os.path.isfile = isfile
    os.path.isdir = isdir
    if os.name == 'nt':
        os.path.realpath = os.path.abspath
    return full_path