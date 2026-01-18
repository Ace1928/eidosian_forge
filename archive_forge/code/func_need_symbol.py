import os
import unittest
from test import support
from test.support import import_helper
def need_symbol(name):
    return unittest.skipUnless(name in ctypes_symbols, '{!r} is required'.format(name))