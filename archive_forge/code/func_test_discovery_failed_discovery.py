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
def test_discovery_failed_discovery(self):
    from test.test_importlib import util
    loader = unittest.TestLoader()
    package = types.ModuleType('package')

    def _import(packagename, *args, **kwargs):
        sys.modules[packagename] = package
        return package
    with unittest.mock.patch('builtins.__import__', _import):
        with import_helper.DirsOnSysPath():
            with util.uncache('package'):
                with self.assertRaises(TypeError) as cm:
                    loader.discover('package')
                self.assertEqual(str(cm.exception), "don't know how to discover from {!r}".format(package))